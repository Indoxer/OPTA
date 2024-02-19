from typing import List

import numpy as np
import torch
from shapely.geometry import Polygon
from tqdm import tqdm

from models import SpaceTransformer
from polygons import RealShapeGenerator, Space


def calc_loss(
    polygons: List[Polygon], turns: np.array, positions: np.array, division_num: int
):
    space = Space(division_num)
    for pol, turn, pos in zip(polygons, turns, positions):
        space.place(pol, turn, pos)

    return space.outliers + space.overlapping


def polygons_to_tensor(polygons: List[Polygon]):
    for pol in polygons:
        
    
    polygons = [[[p for p in pol.exterior.coords]] for pol in polygons]
    
    return torch.tensor()


def train(steps: int, batch_size: int, division_num, n_polygons: int, lr=0.001):
    generator = RealShapeGenerator(scale=0.0025)

    model = SpaceTransformer(8, 2, 48, 4, division_num)

    opt = torch.optim.Adam(model.parameters(), lr)

    last_propability = None

    for step in tqdm(range(steps)):
        opt.zero_grad()

        polygons = [
            polygons_to_tensor(generator.get_next_n_polygons(n_polygons))
            for _ in range(batch_size)
        ]
        polygons = torch.tensor(polygons)

        out = model(polygons)  # [batch_size, n_polygons, 1+scales*2]

        shape = out.shape
        size = shape[-1] // 2

        std, mu = out.split(size)

        shape[-1] = size

        dist = torch.distributions.normal.Normal(mu, std)

        values = dist.rsample()
        turns = values[:, :, 0]
        positions = values[:, :, 1:]

        losses = torch.zeros(batch_size)

        for i in range(batch_size):
            losses[i] = calc_loss(polygons, turns[i], positions[i], division_num)

        propability = torch.exp(dist.log_prob(torch.Tensor(values)))
        loss = torch.log(propability) * losses

        loss = loss.mean()

        print(loss)

        loss.backward()
        opt.step()


def main():
    train(100, 5, 3, 20, 0.001)


if __name__ == "__main__":
    main()
