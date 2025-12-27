import torch
from denoising_diffusion_pytorch import denoising_diffusion_pytorch as base
from tqdm import tqdm
from einops import rearrange, reduce
from functools import partial
import losses

class Diffusion(base.GaussianDiffusion):
    def __init__(self, 
        model, timesteps, sampling_timesteps, sampling_verbose = False, 
        objective = 'pred_v', beta_schedule = 'sigmoid', schedule_fn_kwargs = dict(), 
        custom_schedule = None, loss= None, 
        seed_noise = None, 
    ):
        super().__init__(model, image_size=256, timesteps=timesteps, sampling_timesteps=sampling_timesteps, beta_schedule=beta_schedule, schedule_fn_kwargs=schedule_fn_kwargs if schedule_fn_kwargs else dict())
        self.sampling_verbose = sampling_verbose
        self.objective = objective
        self.custom_schedule = custom_schedule
        self.loss = loss
        self.seed_noise = seed_noise
        
        # for reproducibility
        self.register_buffer('seed_noise_cache_256', torch.zeros(1, 3, 256, 256), persistent=True)
        self.register_buffer('seed_noise_cache_512', torch.zeros(1, 3, 512, 512), persistent=True)

    def p_losses(self, target_noisy, clean_image, refer_noisy, refer_clean, timestep, noise = None, offset_noise_strength = None, *args, **kwargs):
        x_start = target_noisy
        t = timestep
        b, _, h, w = x_start.shape

        noise = base.default(noise, lambda: torch.randn_like(x_start))
        offset_noise_strength = base.default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        model_out = self.model(diffusion_noise=x, clean_image = clean_image, refer_noisy=refer_noisy, refer_clean=refer_clean, time=t)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = torch.nn.functional.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * base.extract(self.loss_weight, t, loss.shape)


        return loss.mean()

    def diffusion_loss(self, noisy_image, clean_image, refer_noisy, refer_clean, *args, **kwargs):
        b, _, h, w, device, = *noisy_image.shape, noisy_image.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        if self.custom_schedule == "sigmoid":
            t = (torch.sigmoid((t - self.num_timesteps / 2) * 0.05) * (self.num_timesteps - 1)).long()
            
        noisy_image = self.normalize(noisy_image)
        clean_image = self.normalize(clean_image)
        refer_noisy = self.normalize(refer_noisy)
        refer_clean = self.normalize(refer_clean)
        return self.p_losses(target_noisy=noisy_image, clean_image=clean_image, refer_noisy=refer_noisy, refer_clean=refer_clean, timestep=t, *args, **kwargs)

    def forward(self, noisy_image, clean_image, refer_noisy, refer_clean, *args, **kwargs):
        loss = 0
        if self.loss["diffusion"]>0:
            loss = loss +  float(self.loss["diffusion"]) * self.diffusion_loss(noisy_image, clean_image, refer_noisy, refer_clean, *args, **kwargs)
        if self.loss["refine"]>0:
            trainable_sample = self.trainable_ddim_sample(
                shape=noisy_image.shape, return_all_timesteps=False, clean_image=clean_image, refer_noisy=refer_noisy, refer_clean=refer_clean, fix_random_seed=True)
            refine_loss = losses.refine_loss(
                trainable_sample, 
                noisy_image, 
                bins=256
            )
            loss = loss + float(self.loss["refine"]) * (refine_loss + torch.nn.functional.mse_loss(trainable_sample, noisy_image))
        return loss

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False, clean_image = None, refer_noisy = None, refer_clean = None):
        model_output = self.model(diffusion_noise=x, time=t, clean_image=clean_image, refer_noisy=refer_noisy, refer_clean=refer_clean)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else base.identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return base.ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True, clean_image = None, refer_noisy = None, refer_clean = None, camera_data = 0):
        preds = self.model_predictions(x, t, x_self_cond, clean_image= clean_image, refer_noisy=refer_noisy, refer_clean=refer_clean)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t: int, x_self_cond = None, clean_image = None, refer_noisy = None, refer_clean = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True, clean_image = clean_image, refer_noisy = refer_noisy, refer_clean = refer_clean)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def p_sample_loop(self, shape, return_all_timesteps = False, clean_image = None, refer_noisy = None, refer_clean = None, fix_random_seed=True):
        batch, device = shape[0], self.device

        if self.seed_noise["random_seed"] == None:
            img = torch.randn(shape, device = device)
            imgs = [img]
        elif self.seed_noise["device"] == "A6000":
            if shape[-2] == shape[-1] == 256:
                img = self.seed_noise_cache_256
            elif shape[-2] == shape[-1] == 512:
                img = self.seed_noise_cache_512
            else:
                raise ValueError(f"Invalid shape: {shape}")
            imgs = [img]
        else:
            torch.manual_seed(self.seed_noise["random_seed"])
            img = torch.randn((1,*shape[1:]), device = self.seed_noise["device"]).cuda()
            img = torch.cat([img]*shape[0], dim=0)
            imgs = [img]
        x_start = None

        loader = reversed(range(0, self.num_timesteps))

        for t in tqdm(loader, desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond, clean_image = clean_image, refer_noisy = refer_noisy, refer_clean = refer_clean)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    def ddim_sample(self, shape, return_all_timesteps = False, clean_image = None, refer_noisy = None, refer_clean = None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1) 
        times = list(reversed(times.int().tolist()))

        time_pairs = list(zip(times[:-1], times[1:]))

        if self.seed_noise["random_seed"] == None:
            img = torch.randn(shape, device = device)
            if return_all_timesteps:
                imgs = [img]
        elif self.seed_noise["device"] == "A6000":
            if shape[-2] == shape[-1] == 256:
                img = self.seed_noise_cache_256
            elif shape[-2] == shape[-1] == 512:
                img = self.seed_noise_cache_512
            else:
                raise ValueError(f"Invalid shape: {shape}")
            if return_all_timesteps:
                imgs = [img]
        else:
            torch.manual_seed(self.seed_noise["random_seed"])
            img = torch.randn((1,*shape[1:]), device = self.seed_noise["device"]).cuda()
            img = torch.cat([img]*shape[0], dim=0)
            if return_all_timesteps:
                imgs = [img]

        x_start = None
        if self.sampling_verbose:
            loader =  tqdm(time_pairs, desc = 'sampling loop time step')
        else:
            loader = time_pairs
        for time, time_next in loader:
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, clean_image = clean_image, refer_noisy = refer_noisy, refer_clean = refer_clean)

            if time_next < 0:
                img = x_start
                if return_all_timesteps:
                    imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            if return_all_timesteps:
                imgs.append(img)

        ret_list = [] if not return_all_timesteps else torch.stack(imgs, dim = 1)
        ret = img
        
        ret = self.unnormalize(ret)
        if return_all_timesteps:
            ret_list = self.unnormalize(ret_list)

        return ret
    
    def trainable_ddim_sample(self, shape, clean_image = None, trainable_step = 2, refer_noisy = None, refer_clean = None, *args, **kwargs):
        if clean_image is not None:
            clean_image = self.normalize(clean_image)
        if refer_noisy is not None:
            refer_noisy = self.normalize(refer_noisy)
        if refer_clean is not None:
            refer_clean = self.normalize(refer_clean)
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1) 
        times = list(reversed(times.int().tolist()))
        
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device = device)

        x_start = None
        if self.sampling_verbose:
            loader =  tqdm(time_pairs, desc = 'sampling loop time step')
        else:
            loader = time_pairs
    
        with torch.no_grad():
            for time, time_next in loader[:-trainable_step]:
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, clean_image = clean_image, refer_noisy = refer_noisy, refer_clean = refer_clean)

                if time_next < 0:
                    img = x_start
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)

                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
            
        for time, time_next in loader[-trainable_step:]:
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, clean_image = clean_image, refer_noisy = refer_noisy, refer_clean = refer_clean)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        ret = self.unnormalize(img)
        return ret

    def sample(self, batch_size = 16, return_all_timesteps = False, clean_image = None, refer_noisy = None, refer_clean = None, *args, **kwargs):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if clean_image is not None:
            clean_image = self.normalize(clean_image)
        if refer_noisy is not None:
            refer_noisy = self.normalize(refer_noisy)
        if refer_clean is not None:
            refer_clean = self.normalize(refer_clean)
        return sample_fn((batch_size, *clean_image.shape[1:]), return_all_timesteps = return_all_timesteps, clean_image=clean_image, refer_noisy = refer_noisy, refer_clean = refer_clean)
