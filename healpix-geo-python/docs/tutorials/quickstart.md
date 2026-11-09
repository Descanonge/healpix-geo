---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

```{code-cell} ipython3
---
editable: true
deletable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
%matplotlib inline
```

+++ {"editable": true, "deletable": true, "slideshow": {"slide_type": ""}}

# Quickstart: `healpix-geo` for the impatient

If you are already familiar with HEALPix, this tutorial quickly presents the various operations available in `healpix-geo`.
For details on HEALPix and `healpix-geo`, see the {doc}`user-guide` or the other tutorials that will go more in depth for each concept.

```{code-cell} ipython3
---
editable: true
deletable: true
slideshow:
  slide_type: ''
---
from healpix_geo.auto import Grid

grid = Grid(level=8, indexing_scheme="nested", ellipsoid="WGS84")
grid
```

+++ {"editable": true, "deletable": true, "slideshow": {"slide_type": ""}}

## First example : Coordinates conversion

The most common task is to convert geographic coordinates into HEALPix cell identifiers.

```{code-cell} ipython3
---
editable: true
deletable: true
slideshow:
  slide_type: ''
---
import numpy as np
from healpix_geo.auto import Grid, lonlat_to_healpix

# Define some points of interest (longitude, latitude in degrees)
paris = (2.3522, 48.8566)
tokyo = (139.6917, 35.6895)
new_york = (-74.0060, 40.7128)

lon = np.array([paris[0], tokyo[0], new_york[0]])
lat = np.array([paris[1], tokyo[1], new_york[1]])

# Convert in HEALPix cells (level 8, nested)
grid = Grid(level=8, indexing_scheme="nested", ellipsoid="WGS84")
ipix = lonlat_to_healpix(lon, lat, grid)

print(f"HEALPix cells (depth={grid.level}):")
print(f"  Paris: {ipix[0]}")
print(f"  Tokyo: {ipix[1]}")
print(f"  New York: {ipix[2]}")
```

+++ {"editable": true, "deletable": true, "slideshow": {"slide_type": ""}}

:::{note}
Cell indices are unique integers that identify each cell. At level 8, each city is in a different cell.
:::

## Refinement level

The `level` parameter (or `depth` when using indexing-specific modules) controls the tiling resolution :

- `level=0` : 12 cells around the world (~6 520 km per cell)
- `level=8` : 786 432 cells (~25 km per cell)
- `level=12` : 201 326 592 cells (~1.6 km per cell)

:::{seealso}
See {ref}`cell-statistics` for a complete table of levels with their resolutions.
:::

```{code-cell} ipython3
from healpix_geo.auto import Grid, lonlat_to_healpix

# Same point at different resolutions
lon, lat = np.array([2.3522]), np.array([48.8566])  # Paris

for level in [0, 4, 8, 12]:
    ipix = lonlat_to_healpix(lon, lat, Grid(level=level, indexing_scheme="nested", ellipsoid="WGS84"))
    print(f"Depth {level:2d}: cell {ipix[0]:10d}")
```

+++ {"editable": true, "deletable": true, "slideshow": {"slide_type": ""}}

:::{tip}
**General rule**: Increasing the `level` by 1 multiplies the number of cells by 4 and divides the cell size by 2.
:::

## Cell geometry

Recover the coordinates of the **center** of a cell:

```{code-cell} ipython3
---
editable: true
deletable: true
slideshow:
  slide_type: ''
---
from healpix_geo.auto import Grid, healpix_to_lonlat, lonlat_to_healpix

paris = (2.3522, 48.8566)
grid = Grid(level=8, indexing_scheme="nested", ellipsoid="WGS84")

# Obtain the cell index
ipix = lonlat_to_healpix(*paris, grid)

# Obtain coordinates of the cell center
lon_center, lat_center = healpix_to_lonlat(ipix, grid)

print(f"Cell center {ipix[0]}:")
print(f"  Longitude: {lon_center[0]:.4f}°")
print(f"  Latitude:  {lat_center[0]:.4f}°")
```

+++ {"editable": true, "deletable": true, "slideshow": {"slide_type": ""}}

:::{note}
The center coordinates may be slightly different from the original coordinates, because we have "rounded" to the nearest cell.
:::

You can also recover the coordinates of the **vertices** of a cell:

```{code-cell} ipython3
---
editable: true
deletable: true
slideshow:
  slide_type: ''
---
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from healpix_geo.auto import Grid, healpix_to_lonlat, vertices

cell_ids = np.arange(32, 42)
grid = Grid(level=8, indexing_scheme="nested", ellipsoid="WGS84")

lon_center, lat_center = healpix_to_lonlat(cell_ids, grid)
lon_vert, lat_vert = vertices(cell_ids, grid)

fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
ax.scatter(lon_center, lat_center)
ax.add_collection(
    PolyCollection(
        np.stack([lon_vert, lat_vert], axis=-1), ec="k", fc="none"
    )
)
```

+++ {"editable": true, "deletable": true, "slideshow": {"slide_type": ""}}

## Research of neighbour

HEALPix makes it easy to find **neighboring** cells:

```{code-cell} ipython3
from healpix_geo.auto import Grid, kth_neighbourhood

ipix = np.array([100])
grid = Grid(level=8, indexing_scheme="nested", ellipsoid="WGS84")
ring = 1  # direct neighbour (distance 1)

neighbours = kth_neighbourhood(ipix, grid, ring=ring)
print(f"Direct neighbourhoods of the cell {ipix[0]}:")
print(neighbours)
```

:::{seealso}
For more details on hierarchy and neighbourhood, see {doc}`../user-guide/hierarchical_indexing`.
:::

## Hierarchic navigation

The next functions are only available for the `nested` indexing scheme, via its module.

```{code-cell} ipython3
from healpix_geo.nested import zoom_to

parent_ipix = 100
parent_depth = 5

# Obtain the 4 children cells
children_ipix = zoom_to(parent_ipix, parent_depth, parent_depth + 1)

print(f"Parent cell:  {parent_ipix} (depth={parent_depth})")
print(f"Children cell: {children_ipix} (depth={parent_depth + 1})")
```

## Sibling

```{code-cell} ipython3
import numpy as np
from healpix_geo.nested import siblings, healpix_to_lonlat

depth = 3

# Choose one cell
ipix = np.array([42], dtype=np.uint64)

# Get siblings (same parent)
sib = siblings(ipix, depth)

print(f"Cell: {ipix[0]} at depth {depth}")
print(f"Siblings: {sib}")
```

Visualisation :

```{code-cell} ipython3
---
tags: [hide-input]
editable: true
deletable: true
slideshow:
  slide_type: ''
---
import matplotlib.pyplot as plt
import numpy as np
from healpix_geo.nested import siblings, healpix_to_lonlat

depth = 3
ipix = np.array([42], dtype=np.uint64)

# Get siblings
sib = siblings(ipix, depth)

# Get coordinates
lon_sib, lat_sib = healpix_to_lonlat(sib, depth, ellipsoid="sphere")
lon_cell, lat_cell = healpix_to_lonlat(ipix, depth, ellipsoid="sphere")

# Plot
fig, ax = plt.subplots(figsize=(8, 4))

# Plot siblings
ax.scatter(lon_sib, lat_sib, s=200, label="Siblings")

# Highlight original cell
ax.scatter(lon_cell, lat_cell, s=250, marker="*", label="Selected cell")

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Siblings at depth {depth}")
ax.legend()
```

+++ {"editable": true, "deletable": true, "slideshow": {"slide_type": ""}}

## Next Steps

Now you know basics, you can explore :

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} HEALPix fundamentals and indexing
:link: healpix
:link-type: doc
:::

:::{grid-item-card} Coordinates and cell geometry
:link: coordinates
:link-type: doc
:::

:::{grid-item-card} Ellipsoids
:link: ellipsoid_basics
:link-type: doc
:::

:::{grid-item-card} Hierarchical indexing
:link: hierarchy
:link-type: doc
:::

:::{grid-item-card} Cover requests
:link: coverage_queries
:link-type: doc
:::

:::{grid-item-card} Working with MOC
:link: working_with_moc
:link-type: doc
:::

:::{grid-item-card} Performance optimisation
:link: performance_optimization
:link-type: doc
:::

::::
