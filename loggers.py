import omegaconf
import torch
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter


class CustomLogger(Logger):
    def __init__(
        self,
        save_dir: str,
        name: str,
        version: str,
        cfg: OmegaConf = None,
        hparams: dict = None,
    ):
        super().__init__()

        self.logs_dir = f"{save_dir}/{name}/{version}"
        self._version = version

        self.writer = SummaryWriter(
            log_dir=self.logs_dir,
        )

        # hparams = {**cfg.model, **cfg.train}

        for key, value in hparams.items():
            if isinstance(value, omegaconf.listconfig.ListConfig):
                hparams[key] = str(value)

        self.hparams = hparams
        self.metrics = {}

    @property
    def name(self):
        return "CustomLogger"

    @property
    def version(self):
        return self._version

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass

    @rank_zero_only
    def update_hyperparams(self, hparams: dict, metrics: dict):
        self.hparams.update(hparams)
        self.metrics.update(metrics)

    @rank_zero_only
    def log_dict(self, params: dict, step: int):
        for key, value in params.items():
            self.writer.add_scalar(key, value, step)

    @rank_zero_only
    def add_images(self, tag: str, images: torch.Tensor, step: int):
        self.writer.add_images(tag, images, step)

    @rank_zero_only
    def finalize(self, status):
        self.writer.add_hparams(self.hparams, self.metrics, run_name=self.logs_dir)
        self.writer.close()
