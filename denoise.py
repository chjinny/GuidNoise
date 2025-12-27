import os
import torch
import numpy as np
import pandas as pd
import argparse
import base64
import torch.nn.functional as F
import math
import cv2
from scipy.io import loadmat
from tqdm import tqdm

from models.dncnn.network_plain import DnCNN

def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.cuda()


@torch.no_grad()
def my_srgb_denoiser(x, model, device='cuda'):
    x_tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)
    denoised_tensor = model(x_tensor)
    denoised_array = (
        denoised_tensor.squeeze().cpu().clamp(0, 1).mul(255.0).byte().permute(1, 2, 0).numpy()
    )

    return denoised_array

def calculate_psnr(img1, img2, border=0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB to YCbCr (PyTorch version)"""
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img


def calculate_ssim_pt(img1, img2, crop_border=0, test_y_channel=False):
    """Calculate SSIM using PyTorch"""
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'

    if crop_border != 0:
        img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img1 = rgb2ycbcr_pt(img1, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img1 = img1.to(torch.float64)
    img2 = img2.to(torch.float64)

    return _ssim_pth(img1 * 255., img2 * 255.)


def _ssim_pth(img1, img2):
    """Calculate SSIM (structural similarity)"""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img1.size(1), 1, 11, 11).to(img1.dtype).to(img1.device)

    mu1 = F.conv2d(img1, window, stride=1, padding=0, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, stride=1, padding=0, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, stride=1, padding=0, groups=img1.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    return ssim_map.mean([1, 2, 3])


def array_to_base64string(x):
    """Convert array to base64 string"""
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string


def run_validation(opt, model, device):
    """Calculate PSNR/SSIM for validation dataset"""
    print('=== Validation Mode ===')
    
    gt_data = loadmat(opt.valid_gt_path)['ValidationGtBlocksSrgb']
    noisy_data = loadmat(opt.valid_noisy_path)['ValidationNoisyBlocksSrgb']
    
    print(f'GT shape: {gt_data.shape}, Noisy shape: {noisy_data.shape}')
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_blocks = gt_data.shape[0] * gt_data.shape[1]
    
    for i in range(gt_data.shape[0]):
        for j in range(gt_data.shape[1]):
            gt_block = gt_data[i, j]
            noisy_block = noisy_data[i, j]
            
            denoised_block = my_srgb_denoiser(noisy_block, model, device)
            
            block_psnr = calculate_psnr(gt_block, denoised_block)
            total_psnr += block_psnr
            
            gt_tensor = (
                torch.from_numpy(gt_block)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .div(255.0)
            )
            denoised_tensor = (
                torch.from_numpy(denoised_block)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .div(255.0)
            )
            
            block_ssim = calculate_ssim_pt(gt_tensor, denoised_tensor)
            ssim_value = block_ssim.item()
            total_ssim += ssim_value
            
            print(f'Block ({i}, {j}) - PSNR: {block_psnr:.2f}, SSIM: {ssim_value:.4f}')
    
    avg_psnr = total_psnr / num_blocks
    avg_ssim = total_ssim / num_blocks
    
    print(f'\nAverage PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}')


def run_benchmark(opt, model, device):
    print('=== Benchmark Mode ===')
    
    input_file = opt.benchmark_path
    key = 'BenchmarkNoisyBlocksSrgb'
    inputs = loadmat(input_file)[key]
    
    print(f'Benchmark shape: {inputs.shape}')
    print(f'Benchmark dtype: {inputs.dtype}')
    
    output_blocks_base64string = []
    
    for i in tqdm(list(range(inputs.shape[0])), desc='Processing benchmark'):
        for j in range(inputs.shape[1]):
            in_block = inputs[i, j, :, :, :]
            out_block = my_srgb_denoiser(in_block, model, device)
            
            assert in_block.shape == out_block.shape
            assert in_block.dtype == out_block.dtype
            
            out_block_base64string = array_to_base64string(out_block)
            output_blocks_base64string.append(out_block_base64string)
    
    print(f'Saving outputs to {opt.output_file}')
    output_df = pd.DataFrame()
    n_blocks = len(output_blocks_base64string)
    print(f'Number of blocks: {n_blocks}')
    output_df['ID'] = np.arange(n_blocks)
    output_df['BLOCK'] = output_blocks_base64string
    
    output_df.to_csv(opt.output_file, index=False)
    print(f'Done. Output saved to: {opt.output_file}')

def main():
    parser = argparse.ArgumentParser(description='SIDD Denoising Inference')
    
    parser.add_argument('--mode', type=str, default='validation', choices=['validation', 'benchmark'],
                        help='Mode: validation or benchmark')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    parser.add_argument('--valid_gt_path', type=str, default='exclude/ValidationGtBlocksSrgb.mat',
                        help='Path to validation GT blocks')
    parser.add_argument('--valid_noisy_path', type=str, default='exclude/ValidationNoisyBlocksSrgb.mat',
                        help='Path to validation noisy blocks')
    
    parser.add_argument('--benchmark_path', type=str, default='exclude/BenchmarkNoisyBlocksSrgb.mat',
                        help='Path to benchmark noisy blocks')
    parser.add_argument('--output_file', type=str, default='SubmitSrgb.csv',
                        help='Output CSV file for benchmark')
    
    opt = parser.parse_args()
    
    print(f'Loading model from: {opt.checkpoint_path}')
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    model = DnCNN(in_nc=3, out_nc=3, nc=64, act_mode='BR').to(device)
    model = load_checkpoint(opt.checkpoint_path, model)
    print('Model loaded successfully')
    
    if opt.mode == 'validation':
        run_validation(opt, model, device)
    elif opt.mode == 'benchmark':
        run_benchmark(opt, model, device)


if __name__ == '__main__':
    main()

