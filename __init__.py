from functools import cache
from typing import Any

import piq
import torch


class PSNR:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "reduction": [
                    "mean",
                    "sum",
                ],
                "convert_to_greyscale": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("psnr",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        data_range: float,
        reduction: str,
        convert_to_greyscale: bool,
    ) -> tuple[float]:
        return (
            piq.psnr(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
                data_range=data_range,
                reduction=reduction,
                convert_to_greyscale=convert_to_greyscale,
            ).item(),
        )


class SSIM:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "kernel_size": (
                    "INTEGER",
                    {
                        "default": 11,
                    },
                ),
                "kernel_sigma": (
                    "FLOAT",
                    {
                        "default": 1.5,
                    },
                ),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "reduction": [
                    "mean",
                    "sum",
                ],
                "full": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "downsample": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
                "k1": (
                    "FLOAT",
                    {
                        "default": 0.01,
                    },
                ),
                "k2": (
                    "FLOAT",
                    {
                        "default": 0.03,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("ssim",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        kernel_size: int,
        kernel_sigma: float,
        data_range: float,
        reduction: str,
        full: bool,
        downsample: bool,
        k1: float,
        k2: float,
    ) -> tuple[float]:
        return (
            piq.ssim(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
                kernel_size=kernel_size,
                kernel_sigma=kernel_sigma,
                data_range=data_range,
                reduction=reduction,
                full=full,
                downsample=downsample,
                k1=k1,
                k2=k2,
            ).item(),
        )


class MSSSIM:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "kernel_size": (
                    "INTEGER",
                    {
                        "default": 11,
                    },
                ),
                "kernel_sigma": (
                    "FLOAT",
                    {
                        "default": 1.5,
                    },
                ),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "reduction": [
                    "mean",
                    "sum",
                ],
                "k1": (
                    "FLOAT",
                    {
                        "default": 0.01,
                    },
                ),
                "k2": (
                    "FLOAT",
                    {
                        "default": 0.03,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("ms_ssim",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        kernel_size: int,
        kernel_sigma: float,
        data_range: float,
        reduction: str,
        k1: float,
        k2: float,
    ) -> tuple[float]:
        return (
            piq.multi_scale_ssim(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
                kernel_size=kernel_size,
                kernel_sigma=kernel_sigma,
                data_range=data_range,
                reduction=reduction,
                k1=k1,
                k2=k2,
            ).item(),
        )


class IWSSIM:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "kernel_size": (
                    "INTEGER",
                    {
                        "default": 11,
                    },
                ),
                "kernel_sigma": (
                    "FLOAT",
                    {
                        "default": 1.5,
                    },
                ),
                "k1": (
                    "FLOAT",
                    {
                        "default": 0.01,
                    },
                ),
                "k2": (
                    "FLOAT",
                    {
                        "default": 0.03,
                    },
                ),
                "parent": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
                "blk_size": (
                    "INTEGER",
                    {
                        "default": 3,
                    },
                ),
                "sigma_nsq": (
                    "FLOAT",
                    {
                        "default": 0.4,
                    },
                ),
                "reduction": [
                    "mean",
                    "sum",
                ],
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("iw_ssim",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        data_range: float,
        kernel_size: int,
        kernel_sigma: float,
        k1: float,
        k2: float,
        parent: bool,
        blk_size: int,
        sigma_nsq: float,
        reduction: str,
    ) -> tuple[float]:
        return (
            piq.information_weighted_ssim(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
                data_range=data_range,
                kernel_size=kernel_size,
                kernel_sigma=kernel_sigma,
                k1=k1,
                k2=k2,
                parent=parent,
                blk_size=blk_size,
                sigma_nsq=sigma_nsq,
                reduction=reduction,
            ).item(),
        )


class VIFp:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "sigma_n_sq": (
                    "FLOAT",
                    {
                        "default": 2.0,
                    },
                ),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "reduction": [
                    "mean",
                    "sum",
                ],
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("vifp",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        sigma_n_sq: float,
        data_range: float,
        reduction: str,
    ) -> tuple[float]:
        return (
            piq.vif_p(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
                sigma_n_sq=sigma_n_sq,
                data_range=data_range,
                reduction=reduction,
            ).item(),
        )


class FSIM:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "reduction": [
                    "mean",
                    "sum",
                ],
                "chromatic": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
                "scales": (
                    "INTEGER",
                    {
                        "default": 4,
                    },
                ),
                "orientations": (
                    "INTEGER",
                    {
                        "default": 4,
                    },
                ),
                "min_length": (
                    "INTEGER",
                    {
                        "default": 6,
                    },
                ),
                "mult": (
                    "INTEGER",
                    {
                        "default": 2,
                    },
                ),
                "sigma_f": (
                    "FLOAT",
                    {
                        "default": 0.55,
                    },
                ),
                "delta_theta": (
                    "FLOAT",
                    {
                        "default": 1.2,
                    },
                ),
                "k": (
                    "FLOAT",
                    {
                        "default": 2.0,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("fsim",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        data_range: float,
        reduction: str,
        chromatic: bool,
        scales: int,
        orientations: int,
        min_length: int,
        mult: int,
        sigma_f: float,
        delta_theta: float,
        k: float,
    ) -> tuple[float]:
        return (
            piq.fsim(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
                data_range=data_range,
                reduction=reduction,
                chromatic=chromatic,
                scales=scales,
                orientations=orientations,
                min_length=min_length,
                mult=mult,
                sigma_f=sigma_f,
                delta_theta=delta_theta,
                k=k,
            ).item(),
        )


class SRSIM:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "reduction": [
                    "mean",
                    "sum",
                ],
                "chromatic": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 0.25,
                    },
                ),
                "kernel_size": (
                    "INTEGER",
                    {
                        "default": 3,
                    },
                ),
                "sigma": (
                    "FLOAT",
                    {
                        "default": 3.8,
                    },
                ),
                "gaussian_size": (
                    "INTEGER",
                    {
                        "default": 10,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("srsim",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        data_range: float,
        reduction: str,
        chromatic: bool,
        scale: float,
        kernel_size: int,
        sigma: float,
        gaussian_size: int,
    ) -> tuple[float]:
        return (
            piq.srsim(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
                data_range=data_range,
                reduction=reduction,
                chromatic=chromatic,
                scale=scale,
                kernel_size=kernel_size,
                sigma=sigma,
                gaussian_size=gaussian_size,
            ).item(),
        )


class GMSD:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "reduction": [
                    "mean",
                    "sum",
                ],
                "t": (
                    "FLOAT",
                    {
                        "default": 170 / (255**2),
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("gmsd",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        data_range: float,
        reduction: str,
        t: float,
    ) -> tuple[float]:
        return (
            piq.gmsd(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
                data_range=data_range,
                reduction=reduction,
                t=t,
            ).item(),
        )


class MSGMSD:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "reduction": [
                    "mean",
                    "sum",
                ],
                "chromatic": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "alpha": (
                    "FLOAT",
                    {
                        "default": 0.5,
                    },
                ),
                "beta1": (
                    "FLOAT",
                    {
                        "default": 0.01,
                    },
                ),
                "beta2": (
                    "FLOAT",
                    {
                        "default": 0.32,
                    },
                ),
                "beta3": (
                    "FLOAT",
                    {
                        "default": 15.0,
                    },
                ),
                "t": (
                    "FLOAT",
                    {
                        "default": 170.0,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("msgmsd",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        data_range: float,
        reduction: str,
        chromatic: bool,
        alpha: float,
        beta1: float,
        beta2: float,
        beta3: float,
        t: float,
    ) -> tuple[float]:
        return (
            piq.multi_scale_gmsd(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
                data_range=data_range,
                reduction=reduction,
                chromatic=chromatic,
                alpha=alpha,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                t=t,
            ).item(),
        )


class DSS:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "reduction": [
                    "mean",
                    "sum",
                ],
                "dct_size": (
                    "INTEGER",
                    {
                        "default": 8,
                    },
                ),
                "sigma_weight": (
                    "FLOAT",
                    {
                        "default": 1.55,
                    },
                ),
                "kernel_size": (
                    "INTEGER",
                    {
                        "default": 3,
                    },
                ),
                "sigma_similarity": (
                    "FLOAT",
                    {
                        "default": 1.5,
                    },
                ),
                "percentile": (
                    "FLOAT",
                    {
                        "default": 0.05,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("dss",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        data_range: float,
        reduction: str,
        dct_size: int,
        sigma_weight: float,
        kernel_size: int,
        sigma_similarity: float,
        percentile: float,
    ) -> tuple[float]:
        return (
            piq.dss(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
                data_range=data_range,
                reduction=reduction,
                dct_size=dct_size,
                sigma_weight=sigma_weight,
                kernel_size=kernel_size,
                sigma_similarity=sigma_similarity,
                percentile=percentile,
            ).item(),
        )


@cache
def get_content_loss(
    feature_extractor: str = "vgg16",
    replace_pooling: bool = False,
    distance: str = "mse",
    reduction: str = "mean",
    normalize_features: bool = False,
) -> piq.ContentLoss:
    return piq.ContentLoss(
        feature_extractor=feature_extractor,
        replace_pooling=replace_pooling,
        distance=distance,
        reduction=reduction,
        normalize_features=normalize_features,
    )


class ContentScore:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "feature_extractor": [
                    "vgg16",
                    "vgg19",
                ],
                "replace_pooling": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "distance": [
                    "mse",
                    "mae",
                ],
                "reduction": [
                    "mean",
                    "sum",
                ],
                "normalize_features": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("content_loss",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        feature_extractor: str,
        replace_pooling: bool,
        distance: str,
        reduction: str,
        normalize_features: bool,
    ) -> tuple[float]:
        return (
            get_content_loss(
                feature_extractor=feature_extractor,
                replace_pooling=replace_pooling,
                distance=distance,
                reduction=reduction,
                normalize_features=normalize_features,
            )(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
            ).item(),
        )


@cache
def get_style_loss(
    feature_extractor: str = "vgg16",
    replace_pooling: bool = False,
    distance: str = "mse",
    reduction: str = "mean",
    normalize_features: bool = False,
) -> piq.StyleLoss:
    return piq.StyleLoss(
        feature_extractor=feature_extractor,
        replace_pooling=replace_pooling,
        distance=distance,
        reduction=reduction,
        normalize_features=normalize_features,
    )


class StyleScore:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "feature_extractor": [
                    "vgg16",
                    "vgg19",
                ],
                "replace_pooling": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "distance": [
                    "mse",
                    "mae",
                ],
                "reduction": [
                    "mean",
                    "sum",
                ],
                "normalize_features": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("style_loss",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        feature_extractor: str,
        replace_pooling: bool,
        distance: str,
        reduction: str,
        normalize_features: bool,
    ) -> tuple[float]:
        return (
            get_style_loss(
                feature_extractor=feature_extractor,
                replace_pooling=replace_pooling,
                distance=distance,
                reduction=reduction,
                normalize_features=normalize_features,
            )(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
            ).item(),
        )


class HaarPSI:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "scales": (
                    "INTEGER",
                    {
                        "default": 3,
                    },
                ),
                "subsample": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
                "c": (
                    "FLOAT",
                    {
                        "default": 30.0,
                    },
                ),
                "alpha": (
                    "FLOAT",
                    {
                        "default": 4.2,
                    },
                ),
                "reduction": [
                    "mean",
                    "sum",
                ],
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("haar_psi",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        data_range: float,
        scales: int,
        subsample: bool,
        c: float,
        alpha: float,
        reduction: str,
    ) -> tuple[float]:
        return (
            piq.haarpsi(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
                data_range=data_range,
                scales=scales,
                subsample=subsample,
                c=c,
                alpha=alpha,
                reduction=reduction,
            ).item(),
        )


class MDSI:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "reduction": [
                    "mean",
                    "sum",
                ],
                "c1": (
                    "FLOAT",
                    {
                        "default": 140.0,
                    },
                ),
                "c2": (
                    "FLOAT",
                    {
                        "default": 55.0,
                    },
                ),
                "c3": (
                    "FLOAT",
                    {
                        "default": 550.0,
                    },
                ),
                "combination": [
                    "sum",
                    "mult",
                ],
                "alpha": (
                    "FLOAT",
                    {
                        "default": 0.6,
                    },
                ),
                "beta": (
                    "FLOAT",
                    {
                        "default": 0.1,
                    },
                ),
                "gamma": (
                    "FLOAT",
                    {
                        "default": 0.2,
                    },
                ),
                "rho": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "q": (
                    "FLOAT",
                    {
                        "default": 0.25,
                    },
                ),
                "o": (
                    "FLOAT",
                    {
                        "default": 0.25,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("mdsi",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        data_range: float,
        reduction: str,
        c1: float,
        c2: float,
        c3: float,
        combination: str,
        alpha: float,
        beta: float,
        gamma: float,
        rho: float,
        q: float,
        o: float,
    ) -> tuple[float]:
        return (
            piq.mdsi(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
                data_range=data_range,
                reduction=reduction,
                c1=c1,
                c2=c2,
                c3=c3,
                combination=combination,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                rho=rho,
                q=q,
                o=o,
            ).item(),
        )


@cache
def get_lpips_loss(
    replace_pooling: bool,
    distance: str,
    reduction: str,
):
    return piq.LPIPS(
        replace_pooling=replace_pooling,
        distance=distance,
        reduction=reduction,
    )


class LPIPS:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "replace_pooling": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "distance": [
                    "mse",
                    "mae",
                ],
                "reduction": [
                    "mean",
                    "sum",
                ],
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("lpips",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        replace_pooling: bool,
        distance: str,
        reduction: str,
    ) -> tuple[float]:
        return (
            get_lpips_loss(
                replace_pooling=replace_pooling,
                distance=distance,
                reduction=reduction,
            )(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
            ).item(),
        )


@cache
def get_pieapp_loss(
    reduction: str,
    data_range: float,
    stride: int,
    enable_grad: bool,
):
    return piq.PieAPP(
        reduction=reduction,
        data_range=data_range,
        stride=stride,
        enable_grad=enable_grad,
    )


class PieAPP:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "reduction": [
                    "mean",
                    "sum",
                ],
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
                "stride": (
                    "INTEGER",
                    {
                        "default": 8,
                    },
                ),
                "enable_grad": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("pieapp",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        reduction: str,
        data_range: float,
        stride: int,
        enable_grad: bool,
    ) -> tuple[float]:
        return (
            get_pieapp_loss(
                reduction=reduction,
                data_range=data_range,
                stride=stride,
                enable_grad=enable_grad,
            )(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
            ).item(),
        )


@cache
def get_dists_loss(
    reduction: str,
):
    return piq.DISTS(
        reduction=reduction,
    )


class DISTS:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "reduction": [
                    "mean",
                    "sum",
                ],
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("dists",)
    FUNCTION = "process"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        reduction: str,
    ) -> tuple[float]:
        return (
            get_dists_loss(
                reduction=reduction,
            )(
                image_a.permute(0, 3, 1, 2),
                image_b.permute(0, 3, 1, 2),
            ).item(),
        )


NODE_CLASS_MAPPINGS: dict[str, Any] = {
    "PSNR": PSNR,
    "LPIPS": LPIPS,
    "MS-SSIM": MSSSIM,
    "SSIM": SSIM,
    "IW-SSIM": IWSSIM,
    "DSS": DSS,
    "ContentScore": ContentScore,
    "StyleScore": StyleScore,
    "HaarPSI": HaarPSI,
    "MDSI": MDSI,
    "VIFp": VIFp,
    "FSIM": FSIM,
    "SRSIM": SRSIM,
    "GMSD": GMSD,
    "MSGMSD": MSGMSD,
    "PieAPP": PieAPP,
    "DISTS": DISTS,
}
