from typing import Protocol, TypedDict, Generic, TypeVar

import numpy as np
import numpy.typing as npt


class SphereDict(TypedDict):
    radius: float


class SphereType(Protocol):
    radius: float


class EllipsoidDict(TypedDict):
    semimajor_axis: float
    inverse_flattening: float


class EllipsoidType(Protocol):
    semimajor_axis: float
    inverse_flattening: float


_SphereLike = SphereDict | SphereType
_EllipsoidLike = EllipsoidDict | EllipsoidType

EllipsoidLike = str | _SphereLike | _EllipsoidLike

DepthType = int | npt.NDArray[np.uint8]

_ScalarT = TypeVar("_ScalarT", bound=np.generic)
Array1D = np.ndarray[tuple[int], np.dtype[_ScalarT]]
Array2D = np.ndarray[tuple[int, int], np.dtype[_ScalarT]]
