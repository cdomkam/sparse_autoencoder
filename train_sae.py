import os

from sparse_autoencoder import (
    ActivationResamplerHyperparameters,
    AutoencoderHyperparameters,
    Hyperparameters,
    LossHyperparameters,
    Method,
    OptimizerHyperparameters,
    Parameter,
    PipelineHyperparameters,
    SourceDataHyperparameters,
    SourceModelHyperparameters,
    SweepConfig,
    sweep,
)

os.environ["WANDB_NOTEBOOK_NAME"] = "demo.ipynb"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_gpt_small_mlp_layers(
    expansion_factor: int = 4,
    n_layers: int = 12,
) -> None:
    """Run a new sweep experiment on GPT 2 Small's MLP layers.

    Args:
        expansion_factor: Expansion factor for the autoencoder.
        n_layers: Number of layers to train on. Max is 12.

    """
    sweep_config = SweepConfig(
        parameters=Hyperparameters(
            loss=LossHyperparameters(
                l1_coefficient=Parameter(max=0.03, min=0.008),
            ),
            optimizer=OptimizerHyperparameters(
                lr=Parameter(max=0.001, min=0.00001),
            ),
            source_model=SourceModelHyperparameters(
                name=Parameter("gpt2"),
                cache_names=Parameter(
                    [f"blocks.{layer}.hook_mlp_out" for layer in range(n_layers)]
                ),
                hook_dimension=Parameter(768),
            ),
            source_data=SourceDataHyperparameters(
                dataset_path=Parameter("alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"),
                # context_size=Parameter(256),
                context_size=Parameter(4),
                pre_tokenized=Parameter(value=True),
                pre_download=Parameter(value=False),  # Default to streaming the dataset
            ),
            autoencoder=AutoencoderHyperparameters(
                expansion_factor=Parameter(value=expansion_factor)
            ),
            pipeline=PipelineHyperparameters(
                max_activations=Parameter(12),
                # checkpoint_frequency=Parameter(100_000_000),
                checkpoint_frequency=Parameter(12),
                # validation_frequency=Parameter(100_000_000),
                validation_frequency=Parameter(12),
                max_store_size=Parameter(12),
            ),
            activation_resampler=ActivationResamplerHyperparameters(
                resample_interval=Parameter(20),
                n_activations_activity_collate=Parameter(10),
                threshold_is_dead_portion_fires=Parameter(1e-6),
                max_n_resamples=Parameter(4),
            ),
        ),
        method=Method.RANDOM,
    )

    sweep(sweep_config=sweep_config)

def main() -> None:
    train_gpt_small_mlp_layers() 

if __name__=="__main__":
    main()