from audioop import avg
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from shapely.geometry import Polygon
from torchvision.transforms import ToTensor
from tqdm import tqdm

from loggers import CustomLogger
from models import SpaceTransformer
from plotting import show_in_pyplot
from polygons import RealShapeGenerator, Space


class Trainer:
    def __init__(
        self,
        model,
        lr=0.001,
        division_num=2,
        scale=0.0025,
        grad_accumulation_steps=1,
        avg_window=20,
    ):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr)
        self.logger = CustomLogger("logs", "test", "0", hparams={"lr": lr})
        self.generator = RealShapeGenerator(scale=scale)
        self.division_num = division_num
        self.grad_accumulation_steps = grad_accumulation_steps
        self.avg_reward = 0
        self.avg_loss = 0
        self.avg_std = 0
        self.avg_window = avg_window

    def calc_reward(
        self,
        polygons: List[Polygon],
        turns: torch.Tensor,
        positions: torch.Tensor,
    ):
        space = Space(self.division_num)
        for pol, turn, pos in zip(polygons, turns, positions):
            space.place(pol, turn, pos)

        area = 0
        for pol in space.polygons:
            area += pol.area

        return area - (space.outliers + space.overlapping)

    def log_plot_polygons(
        self, polygons: List[Polygon], turns: np.array, positions: np.array, step: int
    ):
        s = Space(self.division_num)
        for pol, turn, pos in zip(polygons, turns, positions):
            s.place(pol, turn, pos)
        buf = show_in_pyplot(s.polygons, s.main, show=False)
        img = PIL.Image.open(buf)
        img = ToTensor()(img).unsqueeze(0)
        self.logger.add_images("polygons", img, step)

    def get_next(self, n_polygons: int):
        polygons = self.generator.get_next_n_polygons(n_polygons)
        tpolygons = [
            torch.tensor([p for p in pol.exterior.coords], device="cuda")
            for pol in polygons
        ]

        return polygons, tpolygons

    def step(self, polygons, tpolygons, step: int):
        turns, positions, log_prob, std = self.model(tpolygons)

        reward = self.calc_reward(polygons, turns, positions)

        loss = -log_prob * reward

        loss = loss.mean()

        self.avg_loss -= self.avg_loss / self.avg_window
        self.avg_loss += loss / self.avg_window

        loss = loss / self.grad_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

        self.avg_reward -= self.avg_reward / self.avg_window
        self.avg_reward += reward / self.avg_window
        self.avg_std -= self.avg_std / self.avg_window
        self.avg_std += std.mean() / self.avg_window

        self.logger.log_dict(
            {
                "metrics/avg_reward": self.avg_reward,
                "metrics/avg_loss": self.avg_loss,
                "metrics/avg_std": self.avg_std,
            },
            step,
        )

        if step % 500 == 0:
            self.log_plot_polygons(polygons, turns, positions, step)

        if step % self.grad_accumulation_steps == 0:
            self.opt.step()
            self.opt.zero_grad()

    def fit(self, steps: int, n_polygons: int):
        for step in tqdm(range(steps)):
            polygons, tpolygons = self.get_next(n_polygons)
            self.step(polygons, tpolygons, step)


def train():
    division_num = 1

    model = SpaceTransformer(8, 2, 48, 4, division_num)
    model.to("cuda")

    trainer = Trainer(
        model,
        lr=0.001,
        division_num=division_num,
        grad_accumulation_steps=5,
        avg_window=150,
    )
    trainer.fit(50000, 10)


if __name__ == "__main__":
    train()
