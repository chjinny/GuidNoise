import importlib
import yaml

def load_model(model_name, type_name):
    dir = f"models.{model_name}"
    config_path = f"models/{model_name}/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)[type_name]
    model = getattr(importlib.import_module(f"{dir}.model"), model_name)(**config)
    return model, config