from functools import cache
from typing import Any

import piq
import torch


class PSNR:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_a": (
                    "IMAGE",
                    {
                        "tooltip": "Input image",
                    },
                ),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image",
                    },
                ),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Maximum value range of images",
                    },
                ),
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                    {
                        "tooltip": "Reduction method",
                    },
                ),
                "convert_to_greyscale": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Convert images to YIQ format and compute PSNR only on luminance channel",
                    },
                ),
            },
        }

    DESCRIPTION = "Measures the peak signal-to-noise ratio between images. Higher values indicate better quality."
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("psnr",)
    OUTPUT_TOOLTIPS = ("Peak Signal-to-Noise Ratio",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
                "image_a": (
                    "IMAGE",
                    {
                        "tooltip": "Input image",
                    },
                ),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image",
                    },
                ),
                "kernel_size": (
                    "INTEGER",
                    {
                        "default": 11,
                        "tooltip": "Size of the Gaussian kernel",
                    },
                ),
                "kernel_sigma": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "tooltip": "Standard deviation of the Gaussian kernel",
                    },
                ),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Maximum value range of images",
                    },
                ),
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                    {
                        "tooltip": "Reduction method",
                    },
                ),
                "downsample": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                    {
                        "tooltip": "Whether to perform downsampling",
                    },
                ),
                "k1": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "tooltip": "First stability constant",
                    },
                ),
                "k2": (
                    "FLOAT",
                    {
                        "default": 0.03,
                        "tooltip": "Second stability constant",
                    },
                ),
            },
        }

    DESCRIPTION = "Measures structural similarity between images, accounting for luminance, contrast, and structure."
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("ssim",)
    OUTPUT_TOOLTIPS = ("Structural Similarity Index",)
    FUNCTION = "process"
    CATEGORY = "piq"

    def process(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        kernel_size: int,
        kernel_sigma: float,
        data_range: float,
        reduction: str,
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
                "image_a": (
                    "IMAGE",
                    {
                        "tooltip": "Input image",
                    },
                ),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image",
                    },
                ),
                "kernel_size": (
                    "INTEGER",
                    {
                        "default": 11,
                        "tooltip": "Size of the Gaussian kernel",
                    },
                ),
                "kernel_sigma": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "tooltip": "Standard deviation of the Gaussian kernel",
                    },
                ),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Maximum value range of images",
                    },
                ),
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                    {
                        "tooltip": "Reduction method",
                    },
                ),
                "k1": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "tooltip": "First stability constant",
                    },
                ),
                "k2": (
                    "FLOAT",
                    {
                        "default": 0.03,
                        "tooltip": "Second stability constant",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("ms_ssim",)
    OUTPUT_TOOLTIPS = ("Multi-Scale Structural Similarity Index",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
                "image_a": (
                    "IMAGE",
                    {
                        "tooltip": "Input image",
                    },
                ),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image",
                    },
                ),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Maximum value range of images",
                    },
                ),
                "kernel_size": (
                    "INTEGER",
                    {
                        "default": 11,
                        "tooltip": "Size of the Gaussian kernel",
                    },
                ),
                "kernel_sigma": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "tooltip": "Standard deviation of the Gaussian kernel",
                    },
                ),
                "k1": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "tooltip": "First stability constant",
                    },
                ),
                "k2": (
                    "FLOAT",
                    {
                        "default": 0.03,
                        "tooltip": "Second stability constant",
                    },
                ),
                "parent": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use parent weights",
                    },
                ),
                "blk_size": (
                    "INTEGER",
                    {
                        "default": 3,
                        "tooltip": "Block size for information weighting",
                    },
                ),
                "sigma_nsq": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "tooltip": "Noise variance",
                    },
                ),
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                    {
                        "tooltip": "Reduction method",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("iw_ssim",)
    OUTPUT_TOOLTIPS = ("Information-Weighted Structural Similarity Index",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
                "image_a": (
                    "IMAGE",
                    {
                        "tooltip": "Input image",
                    },
                ),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image",
                    },
                ),
                "sigma_n_sq": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "tooltip": "Noise variance",
                    },
                ),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Maximum value range of images",
                    },
                ),
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                    {
                        "tooltip": "Reduction method",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("vifp",)
    OUTPUT_TOOLTIPS = ("Visual Information Fidelity",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
                "image_a": (
                    "IMAGE",
                    {
                        "tooltip": "Input image",
                    },
                ),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image",
                    },
                ),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Maximum value range of images",
                    },
                ),
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                    {
                        "tooltip": "Reduction method",
                    },
                ),
                "chromatic": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Whether to include color features",
                    },
                ),
                "scales": (
                    "INTEGER",
                    {
                        "default": 4,
                        "tooltip": "Number of wavelet scales",
                    },
                ),
                "orientations": (
                    "INTEGER",
                    {
                        "default": 4,
                        "tooltip": "Number of filter orientations",
                    },
                ),
                "min_length": (
                    "INTEGER",
                    {
                        "default": 6,
                        "tooltip": "Minimum filter length",
                    },
                ),
                "mult": (
                    "INTEGER",
                    {
                        "default": 2,
                        "tooltip": "Scale multiplication factor",
                    },
                ),
                "sigma_f": (
                    "FLOAT",
                    {
                        "default": 0.55,
                        "tooltip": "Frequency spread",
                    },
                ),
                "delta_theta": (
                    "FLOAT",
                    {
                        "default": 1.2,
                        "tooltip": "Angular interval between filter orientations",
                    },
                ),
                "k": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "tooltip": "Scaling factor",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("fsim",)
    OUTPUT_TOOLTIPS = ("Feature Similarity Index",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                ),
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
    OUTPUT_TOOLTIPS = ("Spectral Residual based Similarity",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
                "image_a": (
                    "IMAGE",
                    {
                        "tooltip": "Input image",
                    },
                ),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image",
                    },
                ),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Maximum value range of images",
                    },
                ),
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                    {
                        "tooltip": "Reduction method",
                    },
                ),
                "t": (
                    "FLOAT",
                    {
                        "default": 170 / (255**2),
                        "tooltip": "Regularization constant",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("gmsd",)
    OUTPUT_TOOLTIPS = ("Gradient Magnitude Similarity Deviation",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
                "image_a": (
                    "IMAGE",
                    {
                        "tooltip": "Input image",
                    },
                ),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image",
                    },
                ),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Maximum value range of images",
                    },
                ),
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                    {
                        "tooltip": "Reduction method",
                    },
                ),
                "chromatic": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Whether to include color features",
                    },
                ),
                "alpha": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "tooltip": "Scale weighting parameter",
                    },
                ),
                "beta1": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "tooltip": "First regularization parameter",
                    },
                ),
                "beta2": (
                    "FLOAT",
                    {
                        "default": 0.32,
                        "tooltip": "Second regularization parameter",
                    },
                ),
                "beta3": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "tooltip": "Third regularization parameter",
                    },
                ),
                "t": (
                    "FLOAT",
                    {
                        "default": 170.0,
                        "tooltip": "Regularization constant",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("msgmsd",)
    OUTPUT_TOOLTIPS = ("Multi-Scale Gradient Magnitude Similarity Deviation",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
                "image_a": (
                    "IMAGE",
                    {
                        "tooltip": "Input image",
                    },
                ),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image",
                    },
                ),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Maximum value range of images",
                    },
                ),
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                    {
                        "tooltip": "Reduction method",
                    },
                ),
                "dct_size": (
                    "INTEGER",
                    {
                        "default": 8,
                        "tooltip": "DCT block size",
                    },
                ),
                "sigma_weight": (
                    "FLOAT",
                    {
                        "default": 1.55,
                        "tooltip": "Standard deviation for weighting",
                    },
                ),
                "kernel_size": (
                    "INTEGER",
                    {
                        "default": 3,
                        "tooltip": "Size of the kernel",
                    },
                ),
                "sigma_similarity": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "tooltip": "Standard deviation for similarity",
                    },
                ),
                "percentile": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "tooltip": "Percentile for coefficient selection",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("dss",)
    OUTPUT_TOOLTIPS = ("Deep Spatial-Spectral Score",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
                "feature_extractor": (
                    [
                        "vgg16",
                        "vgg19",
                    ],
                ),
                "replace_pooling": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "distance": (
                    [
                        "mse",
                        "mae",
                    ],
                ),
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                ),
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
    OUTPUT_TOOLTIPS = ("Content Loss Based on Deep Features",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
                "feature_extractor": (
                    [
                        "vgg16",
                        "vgg19",
                    ],
                ),
                "replace_pooling": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "distance": (
                    [
                        "mse",
                        "mae",
                    ],
                ),
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                ),
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
    OUTPUT_TOOLTIPS = ("Style Loss Based on Deep Features",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
                "image_a": (
                    "IMAGE",
                    {
                        "tooltip": "Input image",
                    },
                ),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image",
                    },
                ),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Maximum value range of images",
                    },
                ),
                "scales": (
                    "INTEGER",
                    {
                        "default": 3,
                        "tooltip": "Number of wavelet scales",
                    },
                ),
                "subsample": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Whether to perform subsampling",
                    },
                ),
                "c": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "tooltip": "Stability constant",
                    },
                ),
                "alpha": (
                    "FLOAT",
                    {
                        "default": 4.2,
                        "tooltip": "Weighting factor",
                    },
                ),
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                    {
                        "tooltip": "Reduction method",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("haar_psi",)
    OUTPUT_TOOLTIPS = ("Haar Perceptual Similarity Index",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                ),
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
                "combination": (
                    [
                        "sum",
                        "mult",
                    ],
                ),
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
    OUTPUT_TOOLTIPS = ("Mean Deviation Similarity Index",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
                "image_a": (
                    "IMAGE",
                    {
                        "tooltip": "Input image",
                    },
                ),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image",
                    },
                ),
                "replace_pooling": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Replace max pooling with average pooling",
                    },
                ),
                "distance": (
                    [
                        "mse",
                        "mae",
                    ],
                    {
                        "tooltip": "Distance metric",
                    },
                ),
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                    {
                        "tooltip": "Reduction method",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("lpips",)
    OUTPUT_TOOLTIPS = ("Learned Perceptual Image Patch Similarity",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
                "image_a": (
                    "IMAGE",
                    {
                        "tooltip": "Input image",
                    },
                ),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image",
                    },
                ),
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                    {
                        "tooltip": "Reduction method",
                    },
                ),
                "data_range": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Maximum value range of images",
                    },
                ),
                "stride": (
                    "INTEGER",
                    {
                        "default": 8,
                        "tooltip": "Stride for patch extraction",
                    },
                ),
                "enable_grad": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable gradient calculation",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("pieapp",)
    OUTPUT_TOOLTIPS = ("Perceptual Image-Error Assessment through Pairwise Preference",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
                "image_a": (
                    "IMAGE",
                    {
                        "tooltip": "Input image",
                    },
                ),
                "image_b": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image",
                    },
                ),
                "reduction": (
                    [
                        "mean",
                        "sum",
                    ],
                    {
                        "tooltip": "Reduction method",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("dists",)
    OUTPUT_TOOLTIPS = ("Deep Image Structure and Texture Similarity",)
    FUNCTION = "process"
    CATEGORY = "piq"

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
