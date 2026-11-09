# Coordinates and cell geometry

This chapter explains how to convert to and from geographical coordinates.

## Coordinates

`healpix-geo` supports two types of geographical coordinates: longitude/latitude and cartesian.

For longitude/latitude, corresponding functions will take in or return:
- `lon`: Longitude in decimal degrees, in [-180, 180], positive towards the East, 0° at Greenwich median.
- `lat`: Latitude in decimal degrees, in [-90, 90], positive towards the North, 0° at the Equator.

For cartesian coordinates, corresponding functions will take in or return `x`, `y`, and `z` in meters. The x-axis points from the Earth center to the point (lon=0, lat=0), the y-axis points to (lon=90, lat=0) and the z-axis points to the North pole.

## Cell geometry

To obtain the coordinates of HEALPix cells **centers** from their indices, use `healpix_to_lonlat` or `healpix_to_cartesian` functions from the appropriate module. For instance:
```python
lon, lat = healpix_geo.auto.healpix_to_lonlat(cell_ids, grid=grid)
```

To obtain the coordinates of HEALPix cells **vertices** from their indices, use `vertices` functions from the appropriate module. For instance:
```python
lon, lat = healpix_geo.auto.vertices(cell_ids, grid=grid)
```
The returned arrays have an additional axis of size 4 appended at the end. For each cell, vertices are returned in order bottom, right, top, left.

## Geographical to healpix

To convert from geographical coordinates to HEALPix cells, use the `lonlat_to_healpix` or `cartesian_to_healpix` functions from the appropriate module. This will return the indices of cells into which the coordinates fall.
For instance:
```python
cell_ids = healpix_geo.auto.lonlat_to_healpix(lon, lat, grid=grid)
```

## See Also

:::{note}
All coordinates passed (`lon`/`lat` or `x`/`y`/`z`) must have the same shape, no broadcasting is done.
:::

For more details, see:

- {doc}`../tutorials/coordinate_conversion` - Conversion Tutorial
