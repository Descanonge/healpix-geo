# User Guide

This guide explains the fundamental concepts of healpix-geo and how to use them effectively.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} HEALPix and indexing schemes
:link: concepts
:link-type: doc

Understanding what HEALPix is, why to use it, and the different indexing schemes it supports.

:::

:::{grid-item-card} Coordinates and cell geometry
:link: coordinates_systems
:link-type: doc

Conversions between geographical coordinates and HEALPix indices. 

:::

:::{grid-item-card} Ellipsoids and geodesy
:link: ellipsoids
:link-type: doc

Earth models, reference ellipsoids, and their impact on your calculations.

:::

:::{grid-item-card} Hierarchical indexing and neighbours
:link: hierarchical_indexing
:link-type: doc

Navigating the hierarchy: parents, children, neighbors.

:::

:::{grid-item-card} Coverage queries
:link: coverage_queries
:link-type: doc

Find all cells that intersect a region: circles, polygons, bounding boxes. etc.

:::

:::{grid-item-card} Performance
:link: performance
:link-type: doc

Optimize your calculations: vectorization, multi-threading, and best practices for processing large volumes.

:::

::::


```{toctree}
---
hidden: true
---
healpix
ellipsoids
coordinates
hierarchical_indexing
coverage_queries
performance
```
