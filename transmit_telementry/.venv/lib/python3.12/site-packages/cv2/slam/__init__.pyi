__all__: list[str] = []

import cv2
import cv2.typing
import typing as _typing


# Enumerations
NOT_INITIALIZED: int
INITIALIZING: int
TRACKING: int
OdometryState = int
"""One of [NOT_INITIALIZED, INITIALIZING, TRACKING]"""



# Classes
class OdometryParams:
    minInitInliers: int
    minInitParallaxDeg: float
    minInitPoints: int
    hfRatioThresh: float
    minGrowthParallaxDeg: float
    essentialRansacThresh: float
    essentialRansacConfidence: float
    pnpReprojThresh: float
    pnpMinInliers: int
    pnpRansacIters: int
    pnpConfidence: float
    motionModelRadius: float
    motionModelRadiusWide: float
    motionModelMinMatches: int
    descProjThresh: float
    opticalFlowMinInliers: int
    kfMinFrames: int
    kfMaxFrames: int
    kfInlierRatio: float
    kfMinInliers: int
    kfRotThreshDeg: float
    kfTransThresh: float
    localMapTopK: int
    localMapNeighborK: int
    localMapRadius: float

    # Functions
    def __init__(self) -> None: ...


class VisualOdometry:
    # Functions
    @classmethod
    @_typing.overload
    def create(cls, detector: cv2.Feature2D, matcher: cv2.DescriptorMatcher, cameraMatrix: cv2.typing.MatLike, distCoeffs: cv2.typing.MatLike | None = ..., params: OdometryParams = ...) -> VisualOdometry: ...
    @classmethod
    @_typing.overload
    def create(cls, detector: cv2.Feature2D, matcher: cv2.DescriptorMatcher, cameraMatrix: cv2.UMat, distCoeffs: cv2.UMat | None = ..., params: OdometryParams = ...) -> VisualOdometry: ...

    @_typing.overload
    def processFrame(self, image: cv2.typing.MatLike) -> bool: ...
    @_typing.overload
    def processFrame(self, image: cv2.UMat) -> bool: ...

    def reset(self) -> None: ...

    def getState(self) -> OdometryState: ...

    def getLastPose(self) -> cv2.typing.Matx44d: ...

    def getNumKeyframes(self) -> int: ...

    def getNumMapPoints(self) -> int: ...

    def getTrajectory(self) -> _typing.Sequence[cv2.typing.Matx44d]: ...

    def getParams(self) -> OdometryParams: ...

    def setParams(self, params: OdometryParams) -> None: ...



