import io
import os

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import MultiPolygon, Polygon


def show_in_pyplot(polygons, last_polygon=None, grid=False, size=1.5, show=True):
    f = plt.figure()
    ax = f.gca()
    ax.set_aspect(1)
    for p in polygons:
        if type(p) is MultiPolygon:
            for m in p:
                patch = PathPatch(make_compound_path(m), alpha=0.5, zorder=2)
                ax.add_patch(patch)
        elif type(p) is Polygon:
            patch = PathPatch(make_compound_path(p), alpha=0.5, zorder=2)
            ax.add_patch(patch)

    if last_polygon is not None:
        patch = PathPatch(
            make_compound_path(last_polygon), alpha=0.5, color="red", zorder=2
        )
        ax.add_patch(patch)

    plt.xlim(-0.5, size)
    plt.ylim(-0.5, size)

    if grid:
        plt.grid()
    if show:
        plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)
    return buf


def make_compound_path(polygon):
    return Path.make_compound_path(
        Path(list(polygon.exterior.coords)),
        *[Path(list(ring.coords)) for ring in polygon.interiors]
    )
