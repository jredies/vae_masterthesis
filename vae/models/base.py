import time
import os
import pathlib
import itertools
import sys
import logging
from multiprocessing import Pool

import torch

from vae.models.simple_vae import create_vae_model
from vae.models.training import train_vae, get_loaders
from vae.utils import exception_hook, model_path
from vae.models.google_path import return_output_folder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run_experiment(
    path: str,
    gamma: float,
):
    n_layers = 3
    geometry = "flat"
    latent_dim_factor = 0.2
    iw_samples = 0

    train_loader, validation_loader, test_loader, dim = get_loaders()

    vae = create_vae_model(
        dim=dim,
        n_layers=n_layers,
        geometry=geometry,
        latent_dim_factor=latent_dim_factor,
    )
    log.info(f"Running iw_samples: {iw_samples}.")
    log.info(f"Save model as {path}.")

    model_name = f"gamma_{gamma}"

    train_vae(
        vae=vae,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        dim=dim,
        model_path=path,
        file_name=model_name,
        loss_type="iwae" if iw_samples > 0 else "standard",
        iw_samples=iw_samples,
        gamma=gamma,
        plateau_patience=7,
        patience=15,
        epochs=400,
        scheduler_type="plateau",
    )

    model_save_path = path / (model_name + ".pth")
    torch.save(vae.state_dict(), model_save_path)


# sys.excepthook = exception_hook


def main():
    path = return_output_folder()

    params = [0.1, 1.0]

    for gamma in params:
        log.info(f"Running experiment with gamma {gamma}.")
        run_experiment(path=path, gamma=gamma)


if __name__ == "__main__":
    main()
