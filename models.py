import torch


class PolygonTransformer(torch.nn.Module):
    def __init__(self, vertex_dim: int, num_heads: int, poly_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(2, vertex_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(vertex_dim, vertex_dim),
            torch.nn.GELU(),
        )
        self.attention1 = torch.nn.MultiheadAttention(
            vertex_dim, num_heads, dropout=0.05
        )
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(vertex_dim, vertex_dim),
            torch.nn.GELU(),
        )
        self.attention2 = torch.nn.MultiheadAttention(
            vertex_dim, num_heads, dropout=0.05
        )
        self.linear3 = torch.nn.Sequential(
            torch.nn.Linear(vertex_dim, poly_dim),
            torch.nn.GELU(),
        )

    def forward(self, x):
        # x [num_points, 2] [x, y]
        x = self.linear1(x)  # [num_points, vertex_dim]
        x = self.attention1(x, x, x)[0]
        x = x + self.linear2(x)
        x = (
            x + self.attention2(x, x, x)[0]
        )  # possible to mix with shape of the space (circle, square, etc.)
        x = self.linear3(x)
        # sum over the points
        x = x.mean(dim=0)  # [poly_dim] maybe we can use a sum or weighted sum
        return x


class SpaceTransformer(torch.nn.Module):
    def __init__(
        self,
        vertex_dim: int,
        vertex_num_heads: int,
        poly_dim: int,
        polygon_num_heads,
        scales: int,
    ) -> None:
        super().__init__()
        self.poly_transformer = PolygonTransformer(
            vertex_dim, vertex_num_heads, poly_dim
        )
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(poly_dim, poly_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(poly_dim, poly_dim),
            torch.nn.GELU(),
        )
        self.attention1 = torch.nn.MultiheadAttention(
            poly_dim, polygon_num_heads, dropout=0.05
        )
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(poly_dim, poly_dim),
            torch.nn.GELU(),
        )
        self.attention2 = torch.nn.MultiheadAttention(
            poly_dim, polygon_num_heads, dropout=0.05
        )
        self.linear3 = torch.nn.Sequential(
            torch.nn.Linear(poly_dim, poly_dim),
            torch.nn.GELU(),
        )
        self.linear4 = torch.nn.Sequential(
            torch.nn.Linear(poly_dim, poly_dim), torch.nn.ReLU()
        )
        calc_out_dim = (1 + scales * 2) * 2
        self.linear5 = torch.nn.Sequential(
            torch.nn.Linear(poly_dim, calc_out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(calc_out_dim, calc_out_dim),
            torch.nn.ReLU(),
        )
        self.poly_dim = poly_dim

        self.f_mu = torch.nn.Sequential(
            torch.nn.Linear(calc_out_dim, calc_out_dim // 2), torch.nn.Sigmoid()
        )
        self.f_log_var = torch.nn.Sequential(
            torch.nn.Linear(calc_out_dim, calc_out_dim // 2), torch.nn.Sigmoid()
        )

    def forward(self, input):
        # input [num_polygons, num_points, 2] [x, y]
        x = []
        for i in range(len(input)):
            x.append(self.poly_transformer(input[i]))

        x = torch.stack(x, dim=0)

        x = self.linear1(x)
        x = self.attention1(x, x, x)[0]
        x = x + self.linear2(x)
        x = x + self.attention2(x, x, x)[0]
        x = x + self.linear3(x)
        x = self.linear4(x)  # for grid encoding we can use a RELU
        x = self.linear5(x)  # [num_polygons, (1+scales*2)*2]

        mu = self.f_mu(x)
        log_var = self.f_log_var(x)
        std = torch.exp(0.5 * log_var)

        dist = torch.distributions.normal.Normal(mu, std)

        values = dist.rsample()
        dvalues = torch.clip(values.clone().detach().cpu(), min=0, max=1)

        turns = dvalues[:, 0]
        positions = dvalues[:, 1:]
        vertex_num = positions.shape[1]
        positions = positions.view(-1, 2, vertex_num // 2)

        turns = turns.numpy()
        positions = positions.numpy()
        log_prob = dist.log_prob(values)
        return turns, positions, log_prob
        # now we have to make encoding over rotation and translation (maybe we can use sinusoidal encoding) first i try grid encoding
