"""Abstract Sparse Autoencoder Model."""
from abc import ABC, abstractmethod
from typing import final

from jaxtyping import Float, Int64
import torch
from torch import Tensor
from torch.nn import Module, Parameter

from sparse_autoencoder.tensor_types import Axis


class AbstractDecoder(Module, ABC):
    """Abstract Decoder Module.

    Typically includes just a :attr:`weight` parameter.
    """

    _learnt_features: int
    """Number of learnt features (inputs to this layer)."""

    _decoded_features: int
    """Number of decoded features (outputs from this layer)."""

    _n_components: int | None

    def __init__(
        self,
        learnt_features: int,
        decoded_features: int,
        n_components: int | None,
    ) -> None:
        """Initialise the decoder.

        Args:
            learnt_features: Number of learnt features in the autoencoder.
            decoded_features: Number of decoded (output) features in the autoencoder.
            n_components: Number of source model components the SAE is trained on.
        """
        super().__init__()
        self._learnt_features = learnt_features
        self._decoded_features = decoded_features
        self._n_components = n_components

    @property
    @abstractmethod
    def weight(
        self,
    ) -> Float[
        Parameter,
        Axis.names(Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE, Axis.LEARNT_FEATURE),
    ]:
        """Weight.

        Each column in the weights matrix (for a specific component) acts as a dictionary vector,
        representing a single basis element in the learned activation space.
        """

    @property
    @abstractmethod
    def reset_optimizer_parameter_details(self) -> list[tuple[Parameter, int]]:
        """Reset optimizer parameter details.

        Details of the parameters that should be reset in the optimizer, when resetting
        dictionary vectors.

        Returns:
            List of tuples of the form `(parameter, axis)`, where `parameter` is the parameter to
            reset (e.g. encoder.weight), and `axis` is the axis of the parameter to reset.
        """

    @abstractmethod
    def forward(
        self,
        x: Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)],
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)]:
        """Forward Pass.

        Args:
            x: Learned activations.

        Returns:
            Decoded activations.
        """

    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset the parameters."""

    @final
    def update_dictionary_vectors(
        self,
        dictionary_vector_indices: Int64[
            Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE_IDX)
        ],
        updated_weights: Float[
            Tensor,
            Axis.names(Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE, Axis.LEARNT_FEATURE_IDX),
        ],
    ) -> None:
        """Update decoder dictionary vectors.

        Updates the dictionary vectors (rows in the weight matrix) with the given values. Typically
        this is used when resampling neurons (dictionary vectors) that have died.

        Args:
            dictionary_vector_indices: Indices of the dictionary vectors to update.
            updated_weights: Updated weights for just these dictionary vectors.
        """
        if dictionary_vector_indices.numel() == 0:
            return

        with torch.no_grad():
            if self._n_components is None:
                self.weight[:, dictionary_vector_indices] = updated_weights
            else:
                for component_idx in range(self._n_components):
                    self.weight[
                        component_idx, :, dictionary_vector_indices[component_idx]
                    ] = updated_weights[component_idx]

    @abstractmethod
    def constrain_weights_unit_norm(self) -> None:
        """Constrain the weights to have unit norm."""
