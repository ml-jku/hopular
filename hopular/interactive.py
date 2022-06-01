import argparse
import math
import os

import torch.cuda
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from hopular.auxiliary.data import DataModule, find_dataset, list_datasets
from hopular.blocks import EmbeddingBlock, Hopular, HopularBlock


def console_entry() -> None:
    """
    Console interface of Hopular.

    :return: None
    """
    arg_parser = argparse.ArgumentParser(
        description=r'A novel deep learning architecture based on continuous modern Hopfield networks' +
                    r' for tackling small tabular datasets.')
    arg_sub_parsers = arg_parser.add_subparsers(dest=r'mode', required=True)

    # Sub-parser for listing all pre-defined datasets.
    list_parser = arg_sub_parsers.add_parser(
        name=r'list',
        help=r'Get more information on implemented entities.')
    list_parser_mode = list_parser.add_mutually_exclusive_group(required=True)
    list_parser_mode.add_argument(
        r'--datasets', action=r'store_true',
        help=r'list available datasets')

    # Sub-parser for optimising Hopular using pre-defined datasets.
    optim_parser = arg_sub_parsers.add_parser(
        name=r'optim',
        help=r'Optimize Hopular.')

    # Dataset-specific arguments.
    optim_parser.add_argument(
        r'--dataset', type=str,
        help=r'substring of dataset name', required=True)

    # Hopular-specific arguments.
    optim_parser.add_argument(
        r'--logging_dir', type=str,
        help=r'directory for storing training and inference logs of Hopular', default=os.getcwd())
    optim_parser.add_argument(
        r'--num_workers', type=int,
        help=r'worker count of the data loaders', default=1)
    optim_parser.add_argument(
        r'--feature_size', type=int,
        help=r'size of the embedding space of a single feature', default=32)
    optim_parser.add_argument(
        r'--hidden_size', type=int,
        help=r'size of the Hopfield association space', default=0)
    optim_parser.add_argument(
        r'--hidden_size_factor', type=float,
        help=r'factor for scaling the size of the Hopfield association space', default=1.0)
    optim_parser.add_argument(
        r'--num_heads', type=int,
        help=r'count of modern Hopfield networks', default=8)
    optim_parser.add_argument(
        r'--scaling_factor', type=float,
        help=r'factor for scaling beta to steer the retrieval type', default=1.0)
    optim_parser.add_argument(
        r'--input_dropout', type=float,
        help=r'probability of embedding dropout regularization', default=0.1)
    optim_parser.add_argument(
        r'--lookup_dropout', type=float,
        help=r'probability of Hopfield lookup dropout regularization', default=0.1)
    optim_parser.add_argument(
        r'--output_dropout', type=float,
        help=r'probability of summarization dropout regularization', default=0.01)
    optim_parser.add_argument(
        r'--memory_ratio', type=float,
        help=r'sub-sample ratio of external memory during training', default=1.0)
    optim_parser.add_argument(
        r'--num_blocks', type=int,
        help=r'count of Hopular blocks, or count of iterative refinement steps', default=8)

    # Datamodule-specific arguments.
    optim_parser.add_argument(
        r'--batch_size', type=int,
        help=r'sample count of a single mini-batch', default=-1)
    optim_parser.add_argument(
        r'--super_sample_factor', type=int,
        help=r'multiplicity of the training set', default=1)
    optim_parser.add_argument(
        r'--noise_probability', type=float,
        help=r'probability of selecting an input feature for the self-supervised loss', default=1.0)
    optim_parser.add_argument(
        r'--mask_probability', type=float,
        help=r'probability of masking out a selected input feature', default=0.025)
    optim_parser.add_argument(
        r'--replace_probability', type=float,
        help=r'probability of replacing a selected input feature with a randomly drawn feature', default=0.175)

    # Optimizer-specific arguments.
    optim_parser.add_argument(
        r'--initial_feature_loss_weight', type=float,
        help=r'initial weighting factor of self-supervised loss', default=1.0)
    optim_parser.add_argument(
        r'--final_feature_loss_weight', type=float,
        help=r'final weighting factor of self-supervised loss', default=0.0)
    optim_parser.add_argument(
        r'--learning_rate', type=float,
        help=r'base step size to be used for parameter updates', default=1e-3)
    optim_parser.add_argument(
        r'--gamma', type=float,
        help=r'decaying factor of learning rate and initial feature loss weight w.r.t. training cycles', default=1.0)
    optim_parser.add_argument(
        r'--beta_one', type=float,
        help=r'first coefficient for the running average computed in the LAMB optimizer', default=0.9)
    optim_parser.add_argument(
        r'--beta_two', type=float,
        help=r'second coefficient for the running averages computed in the LAMB optimizer', default=0.999)
    optim_parser.add_argument(
        r'--weight_decay', type=float,
        help=r'L2 decaying factor of model parameters used during training', default=0.1)
    optim_parser.add_argument(
        r'--lookup_steps', type=int,
        help=r'count of fast weight updates before the slow weight update takes place', default=1)
    optim_parser.add_argument(
        r'--lookup_ratio', type=float,
        help=r'ratio between fast and slow weights steering the slow weight updates', default=0.005)
    optim_parser.add_argument(
        r'--warmup_ratio', type=float,
        help=r'count of steps before learning rate and feature loss weight annealing starts', default=0.0)
    optim_parser.add_argument(
        r'--num_cycles', type=int,
        help=r'count of steps of a single training cycle', default=1)
    optim_parser.add_argument(
        r'--cold_restart', action=r'store_true',
        help=r'apply warmup and do not apply gamma decay at the beginning of a new training cycle')
    optim_parser.add_argument(
        r'--synchronous_weights', action=r'store_false',
        help=r'synchronize fast and slow weights')

    # Trainer-specific arguments.
    optim_parser.add_argument(
        r'--seed', type=int,
        help=r'random state used for initializing and training Hopular', default=1)
    optim_parser.add_argument(
        r'--num_epochs', type=int,
        help=r'count of training epochs', default=5000)
    optim_parser.add_argument(
        r'--gradient_threshold', type=float,
        help=r'L2 gradient norm threshold', default=1.0)
    optim_parser.add_argument(
        r'--validation_delay', type=int,
        help=r'count of epochs between Hopular validations', default=10)
    optim_parser.add_argument(
        r'--fast_mode', action=r'store_true',
        help=r'execute Hopular in fast mode (may introduce additional numeric noise)')
    args = arg_parser.parse_args()

    if args.mode == r'list':

        # List all available pre-defined datasets.
        pre_defined_datasets = list_datasets()
        print(f'Found {len(pre_defined_datasets)} datasets in total:')
        for dataset in pre_defined_datasets:
            print(f'\t{dataset}')

    elif args.mode == r'optim':
        dataset = find_dataset(name=args.dataset)
        split_index, num_splits = 0, 1
        while split_index < num_splits:

            # Prepare data and Hopular modules.
            seed_everything(seed=args.seed)
            data_module = DataModule(
                dataset=dataset(split_index=split_index),
                batch_size=args.batch_size,
                super_sample_factor=args.super_sample_factor,
                noise_probability=args.noise_probability,
                mask_probability=args.mask_probability,
                replace_probability=args.replace_probability,
                num_workers=args.num_workers
            )
            hopular_model = Hopular.from_data_module(

                # Hopular specific arguments.
                data_module=data_module,
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                hidden_size_factor=args.hidden_size_factor,
                num_heads=args.num_heads,
                scaling_factor=args.scaling_factor,
                input_dropout=args.input_dropout,
                lookup_dropout=args.lookup_dropout,
                output_dropout=args.output_dropout,
                memory_ratio=args.memory_ratio,
                num_blocks=args.num_blocks,

                # Optimizer-specific arguments.
                initial_feature_loss_weight=args.initial_feature_loss_weight,
                final_feature_loss_weight=args.final_feature_loss_weight,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                betas=(args.beta_one, args.beta_two),
                weight_decay=args.weight_decay,
                lookup_steps=args.lookup_steps,
                lookup_ratio=args.lookup_ratio,
                warmup_ratio=args.warmup_ratio,
                num_steps_per_cycle=10000,
                cold_restart=args.cold_restart,
                asynchronous_weights=not args.synchronous_weights
            )
            hopular_model.reset_parameters()

            # Prepare trainer instance.
            max_epochs = args.num_epochs * args.num_cycles
            if not args.cold_restart:
                max_epochs -= (math.floor(args.warmup_ratio * args.num_epochs) - 1) * (args.num_cycles - 1)
            num_mini_batches = max(1, int(math.ceil(len(data_module.dataset.split_train) / args.batch_size)))
            hopular_callback = ModelCheckpoint(monitor=r'hp_metric/val', mode=data_module.dataset.checkpoint_mode.value)
            hopular_trainer = Trainer(
                default_root_dir=args.logging_dir,
                max_epochs=max_epochs,
                log_every_n_steps=1,
                check_val_every_n_epoch=args.validation_delay,
                gradient_clip_val=args.gradient_threshold,
                gradient_clip_algorithm=r'norm',
                gpus=1 if torch.cuda.is_available() else 0,
                callbacks=[hopular_callback],
                deterministic=not args.fast_mode,
                accumulate_grad_batches=num_mini_batches
            )

            # Fit and test Hopular instance (testing is done on chosen best model).
            hopular_trainer.fit(model=hopular_model, datamodule=data_module)
            hopular_trainer.test(datamodule=data_module)

            # Fetch total number of splits and adapt current split counter.
            split_index += 1
            if hasattr(data_module.dataset, r'num_splits'):
                num_splits = data_module.dataset.num_splits


if __name__ == r'__main__':
    console_entry()
