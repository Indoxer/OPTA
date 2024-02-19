from typing import List

import numpy as np
import torch
from shapely.geometry import Polygon
from sympy import poly
from tqdm import tqdm

from models import SpaceTransformer
from polygons import RealShapeGenerator, Space


def calc_loss(
    polygons: List[Polygon],
    turns: torch.Tensor,
    positions: torch.Tensor,
    division_num: int,
):
    space = Space(division_num)
    for pol, turn, pos in zip(polygons, turns, positions):
        pos = pos.clone().detach().numpy()
        turn = turn.clone().detach().numpy()
        space.place(pol, turn, pos)

    return space.outliers + space.overlapping


def polygons_to_tensor(polygons: List[Polygon]):
    polygons = [torch.tensor([[p for p in pol.exterior.coords]]) for pol in polygons]

    return polygons


def train(steps: int, batch_size: int, division_num, n_polygons: int, lr=0.001):
    generator = RealShapeGenerator(scale=0.0025)

    model = SpaceTransformer(8, 2, 48, 4, division_num)

    opt = torch.optim.Adam(model.parameters(), lr)

    torch.autograd.set_detect_anomaly(True)

    for step in tqdm(range(steps)):
        opt.zero_grad()

        polygons = generator.get_next_n_polygons(n_polygons)
        tpolygons = polygons_to_tensor(polygons)

        out = model(tpolygons)  # [n_polygons, (1+scales*2)*2]

        size = out.shape[-1] // 2

        std = out[:, :size]
        mu = out[:, size:]

        dist = torch.distributions.normal.Normal(mu, std)

        values = dist.rsample()
        turns = values[:, 0]
        positions = values[:, 1:]
        vertex_num = positions.shape[1]
        positions = positions.view(-1, vertex_num // 2, 2)

        loss = calc_loss(polygons, turns, positions, division_num)

        propability = torch.exp(dist.log_prob(torch.Tensor(values)))
        loss = torch.log(propability) * loss

        loss = loss.mean()

        # print(loss)

        loss.backward()
        opt.step()


def main():
    train(100, 5, 3, 20, 0.001)


if __name__ == "__main__":
    main()
