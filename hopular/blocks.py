import inspect
import math

import numpy as np
import torch
import torch.nn.functional

from collections import namedtuple
from enum import Enum
from fairscale.nn import checkpoint_wrapper
from functools import reduce
from hflayers import Hopfield
from hopular.auxiliary.data import DataModule
from hopular.optim import DelayedScheduler, Lamb, Lookahead
from itertools import chain
from operator import add
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.parsing import get_init_args
from typing import Any, Dict, List, Optional, Tuple


class EmbeddingBlock(LightningModule):
    """
    Block responsible for embedding an input sample in Hopular.
    """

    def __init__(self,
                 input_sizes: List[int],
                 feature_size: int,
                 feature_discrete: torch.Tensor,
                 dropout_probability: float):
        """
        Initialize an embedding block of Hopular.

        :param input_sizes: original feature sizes (class count for discrete features)
        :param feature_size: size of the embedding space of a single feature
        :param feature_discrete: indices of discrete features
        :param dropout_probability: probability of dropout regularization
        """
        super(EmbeddingBlock, self).__init__()
        self.__input_sizes = input_sizes
        self.__feature_size = feature_size
        self.__feature_discrete = feature_discrete
        self.__dropout_probability = dropout_probability
        self.__feature_boundaries = torch.cumsum(torch.as_tensor([0] + self.__input_sizes), dim=0)
        self.__feature_boundaries = (self.__feature_boundaries[:-1], self.__feature_boundaries[1:])

        # Feature-specific embeddings.
        self.feature_embeddings = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(in_features=input_size, out_features=self.__feature_size, bias=True)
        ) for input_position, input_size in enumerate(self.__input_sizes)])

        # Feature-type-specific embeddings.
        self.register_buffer(name=r'feature_types', tensor=torch.zeros(size=(len(input_sizes),), dtype=int))
        if self.__feature_discrete is not None:
            self.feature_types[self.__feature_discrete] = 1
        self.feature_type_embeddings = torch.nn.Embedding(
            num_embeddings=2, embedding_dim=self.__feature_size)

        # Feature-position-specific embeddings.
        self.register_buffer(name=r'feature_positions', tensor=torch.arange(len(self.__input_sizes), dtype=int))
        self.feature_position_embeddings = torch.nn.Embedding(
            num_embeddings=len(self.feature_positions), embedding_dim=self.__feature_size)

        # Feature-specific output projections.
        self.feature_projections = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.__dropout_probability),
            torch.nn.LayerNorm(normalized_shape=self.__feature_size, elementwise_affine=True, eps=1e-12),
            torch.nn.Linear(in_features=self.__feature_size, out_features=self.__feature_size, bias=True)
        ) for _ in enumerate(self.__input_sizes)])

    def reset_parameters(self) -> None:
        """
        Reset parameters of the current embedding block instance.

        :return: None
        """

        # Initialize feature-specific embeddings.
        for module in self.feature_embeddings:
            for layer in filter(lambda _: hasattr(_, r'reset_parameters'), module):
                if type(layer) is torch.nn.Linear:
                    torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                    bound = 1.0 / math.sqrt(torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)[0])
                    torch.nn.init.uniform_(layer.bias, -bound, bound)
                else:
                    layer.reset_parameters()

        # Initialize feature-type-specific embeddings.
        if self.feature_type_embeddings is not None:
            torch.nn.init.kaiming_uniform_(self.feature_type_embeddings.weight, a=math.sqrt(5))

        # Initialize feature-type-specific embeddings.
        if self.feature_position_embeddings is not None:
            torch.nn.init.kaiming_uniform_(self.feature_position_embeddings.weight, a=math.sqrt(5))

        # Initialize feature-specific output projections.
        for module in self.feature_projections:
            for layer in filter(lambda _: hasattr(_, r'reset_parameters'), module):
                if type(layer) is torch.nn.Linear:
                    torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                    bound = 1.0 / math.sqrt(torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)[0])
                    torch.nn.init.uniform_(layer.bias, -bound, bound)
                else:
                    layer.reset_parameters()

    def forward(self,
                input: torch.Tensor):
        """
        Embed input samples.

        :param input: original input sample of pre-processed dataset
        :return: embedded input sample
        """

        # Embed each feature separately.
        feature_iterator = zip(self.feature_embeddings, *self.__feature_boundaries)
        input_embedded = torch.cat(tuple(feature_embedding(
            input[:, feature_begin:feature_end]
        ) for feature_embedding, feature_begin, feature_end in feature_iterator), dim=1)

        # Add feature type and feature position embeddings.
        if self.feature_type_embeddings is not None:
            input_embedded = input_embedded + self.feature_type_embeddings(self.feature_types).view(1, -1)
        if self.feature_position_embeddings is not None:
            input_embedded = input_embedded + self.feature_position_embeddings(self.feature_positions).view(1, -1)
        input_embedded = input_embedded.view(input_embedded.shape[0], len(self.__input_sizes), self.__feature_size)
        return torch.cat(tuple(feature_projection(
            input_embedded[:, feature_index]
        ) for feature_index, feature_projection in enumerate(self.feature_projections)), dim=1)


class SummarizationBlock(LightningModule):
    """
    Block responsible for summarizing the current prediction in Hopular.
    """

    def __init__(self,
                 input_sizes: List[int],
                 feature_size: int,
                 dropout_probability: float):
        """
        Initialize a summarization block.

        :param input_sizes: original feature sizes (class count for discrete features)
        :param feature_size: size of the embedding space of a single feature
        :param dropout_probability: probability of dropout regularization
        """
        super(SummarizationBlock, self).__init__()
        self.__input_sizes = input_sizes
        self.__feature_size = feature_size
        self.__dropout_probability = dropout_probability

        # Feature-specific summarizations.
        self.__input_size = self.__feature_size * len(self.__input_sizes)
        self.feature_summarizations = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.__dropout_probability),
            torch.nn.LayerNorm(normalized_shape=self.__input_size, elementwise_affine=True, eps=1e-12),
            torch.nn.Linear(in_features=self.__input_size, out_features=reduce(add, self.__input_sizes), bias=True)
        )

    def reset_parameters(self) -> None:
        """
        Reset parameters of the current summarization block.

        :return: None
        """

        # Initialize feature-specific summarizations.
        for layer in filter(lambda _: hasattr(_, r'reset_parameters'), self.feature_summarizations):
            if type(layer) is torch.nn.Linear:
                torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                bound = 1.0 / math.sqrt(torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)[0])
                torch.nn.init.uniform_(layer.bias, -bound, bound)
            else:
                layer.reset_parameters()

    def forward(self,
                input: torch.Tensor):
        """
        Summarize current prediction.

        :param input: current prediction of Hopular
        :return: summarized final prediction
        """

        # Embed each feature separately.
        return self.feature_summarizations(input)


class HopfieldBlock(LightningModule):
    """
    Block responsible for memory lookup operations in Hopular.
    """

    def __init__(self,
                 input_size: int,
                 feature_size: int,
                 hidden_size: int,
                 num_heads: int,
                 scaling_factor: float,
                 dropout_probability: float,
                 normalize: bool):
        """
        Initialize a memory lookup block of Hopular.

        :param input_size: size of the embedding space of a complete sample
        :param feature_size: size of the embedding space of a single feature
        :param hidden_size: size of the Hopfield association space
        :param num_heads: count of modern Hopfield networks
        :param scaling_factor: factor for scaling beta to steer the retrieval type
        :param dropout_probability: probability of dropout regularization
        :param normalize: normalize inputs to memory lookup block
        """
        super(HopfieldBlock, self).__init__()
        self.__input_size = input_size
        self.__feature_size = feature_size
        self.__num_features = self.__input_size // self.__feature_size
        assert (self.__num_features * feature_size) == input_size, r'Invalid input/feature shapes specified.'
        self.__hidden_size = hidden_size
        self.__num_heads = num_heads
        self.__scaling_factor = scaling_factor
        self.__dropout_probability = dropout_probability
        self.__normalize = normalize

        # Hopfield-related associations.
        self.hopfield_lookup = Hopfield(
            input_size=self.__input_size,
            hidden_size=self.__hidden_size,
            pattern_size=self.__hidden_size,
            output_size=self.__input_size,
            num_heads=self.__num_heads,
            scaling=self.__scaling_factor / math.sqrt(self.__hidden_size),
            normalize_state_pattern=self.__normalize,
            normalize_state_pattern_affine=self.__normalize,
            normalize_state_pattern_eps=1e-12,
            normalize_stored_pattern=self.__normalize,
            normalize_stored_pattern_affine=self.__normalize,
            normalize_stored_pattern_eps=1e-12,
            normalize_pattern_projection=self.__normalize,
            normalize_pattern_projection_affine=self.__normalize,
            normalize_pattern_projection_eps=1e-12,
            normalize_hopfield_space=False,
            normalize_hopfield_space_affine=False,
            normalize_hopfield_space_eps=1e-12,
            dropout=self.__dropout_probability
        )

    def reset_parameters(self) -> None:
        """
        Reset parameters of the current memory lookup block.

        :return: None
        """

        # Initialize Hopfield-related associations.
        if self.hopfield_lookup.association_core.p_norm_weight is not None:
            torch.nn.init.ones_(self.hopfield_lookup.association_core.p_norm_weight)
            torch.nn.init.zeros_(self.hopfield_lookup.association_core.p_norm_bias)
        if self.hopfield_lookup.association_core.in_proj_weight is not None:
            torch.nn.init.kaiming_uniform_(self.hopfield_lookup.association_core.in_proj_weight, a=math.sqrt(5))
        else:
            for attribute in (r'q_proj_weight', r'k_proj_weight', r'v_proj_weight'):
                parameter = getattr(self.hopfield_lookup.association_core, attribute)
                if parameter is not None:
                    torch.nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
        if self.hopfield_lookup.association_core.in_proj_bias is not None:
            assert self.hopfield_lookup.association_core.q_proj_weight is not None
            bound = 1.0 / math.sqrt(torch.nn.init._calculate_fan_in_and_fan_out(
                self.hopfield_lookup.association_core.q_proj_weight)[0])
            torch.nn.init.uniform_(self.hopfield_lookup.association_core.in_proj_bias, -bound, bound)
        if not self.hopfield_lookup.association_core.disable_out_projection:
            torch.nn.init.kaiming_uniform_(self.hopfield_lookup.association_core.out_proj.weight, a=math.sqrt(5))
            if self.hopfield_lookup.association_core.out_proj.bias is not None:
                bound = 1.0 / math.sqrt(torch.nn.init._calculate_fan_in_and_fan_out(
                    self.hopfield_lookup.association_core.out_proj.weight)[0])
                torch.nn.init.uniform_(self.hopfield_lookup.association_core.out_proj.bias, -bound, bound)
        for attribute in (r'k', r'v'):
            parameter_bias = getattr(self.hopfield_lookup.association_core, f'bias_{attribute}')
            if parameter_bias is not None:
                parameter_weight = getattr(self.hopfield_lookup.association_core, f'{attribute}_proj_weight')
                bound = 1.0 / math.sqrt(torch.nn.init._calculate_fan_in_and_fan_out(parameter_weight)[0])
                torch.nn.init.uniform_(parameter_bias, -bound, bound)

    def forward(self,
                input: torch.Tensor,
                memory: torch.Tensor,
                memory_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform memory lookup.

        :param input: current prediction of Hopular used for lookup
        :param memory: external memory used for lookup
        :param memory_mask: mask for prohibiting specific lookups
        :return: refined current prediction
        """

        # Retrieve patterns using a continuous modern Hopfield network.
        retrieved_patterns = self.hopfield_lookup((memory, input, memory), association_mask=memory_mask)

        # Compute feature-specific projections.
        return input + retrieved_patterns


class HopularBlock(LightningModule):
    """
    Block responsible for iteratively refining the current prediction in Hopular.
    """

    def __init__(self,
                 input_size: int,
                 feature_size: int,
                 hidden_size: int,
                 hidden_size_factor: float,
                 num_heads: int,
                 scaling_factor: float,
                 dropout_probability: float):
        """
        Initialize an iterative refinement block of Hopular.

        :param input_size: size of the embedding space of a complete sample
        :param feature_size: size of the embedding space of a single feature
        :param hidden_size: size of the Hopfield association space
        :param hidden_size_factor: factor for scaling the size of the Hopfield association space
        :param num_heads: count of modern Hopfield networks
        :param scaling_factor: factor for scaling beta to steer the retrieval type
        :param dropout_probability: probability of dropout regularization
        """
        super(HopularBlock, self).__init__()
        self.__input_size = input_size
        self.__feature_size = feature_size
        self.__num_heads = num_heads
        self.__hidden_size_factor = hidden_size_factor
        self.__hidden_size_sample = hidden_size if hidden_size > 0 else max(1, self.__input_size // self.__num_heads)
        self.__hidden_size_feature = hidden_size if hidden_size > 0 else max(1, self.__feature_size // self.__num_heads)

        self.__scaling_factor = scaling_factor
        self.__dropout_probability = dropout_probability
        assert self.__input_size % self.__feature_size == 0, r'Invalid feature size specified!'

        # Sample-sample associations including residual connection.
        self.sample_norm = None
        self.sample_sample_associations = None
        self.sample_norm = torch.nn.LayerNorm(
            normalized_shape=self.__input_size, elementwise_affine=True, eps=1e-12)
        self.sample_sample_associations = checkpoint_wrapper(HopfieldBlock(
            input_size=self.__input_size,
            feature_size=self.__feature_size,
            hidden_size=max(1, int(self.__hidden_size_sample * self.__hidden_size_factor)),
            num_heads=self.__num_heads,
            scaling_factor=self.__scaling_factor,
            dropout_probability=self.__dropout_probability,
            normalize=False))

        # Feature-feature associations including residual connection.
        self.feature_feature_associations = None
        self.feature_norm = torch.nn.LayerNorm(
            normalized_shape=self.__feature_size, elementwise_affine=True, eps=1e-12)
        self.feature_feature_associations = checkpoint_wrapper(HopfieldBlock(
            input_size=self.__feature_size,
            feature_size=self.__feature_size,
            hidden_size=max(1, int(self.__hidden_size_feature * self.__hidden_size_factor)),
            num_heads=self.__num_heads,
            scaling_factor=self.__scaling_factor,
            dropout_probability=self.__dropout_probability,
            normalize=False))

    def reset_parameters(self) -> None:
        """
        Reset parameters of the current iterative refinement block.

        :return: None
        """

        # Initialize sample-sample associations including residual connection.
        if self.sample_sample_associations is not None:
            self.sample_norm.reset_parameters()
            self.sample_sample_associations.reset_parameters()

        # Initialize feature-feature associations including residual connection.
        if self.feature_feature_associations is not None:
            self.feature_norm.reset_parameters()
            self.feature_feature_associations.reset_parameters()

    def forward(self,
                input: torch.Tensor,
                sample_memory: torch.Tensor,
                sample_memory_mask: torch.Tensor,
                feature_memory: torch.Tensor,
                feature_memory_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform a single prediction refinement step.

        :param input: current prediction of Hopular to be refined
        :param sample_memory: external memory populated with the complete training set
        :param sample_memory_mask: mask for prohibiting specific sample lookups
        :param feature_memory: external memory populated with the current original input representation
        :param feature_memory_mask: mask for prohibiting specific feature lookups
        :return: refined current prediction
        """

        # Compute sample-sample interactions and reshape result accordingly.
        interactions = self.sample_sample_associations(
            self.sample_norm(input),
            memory=self.sample_norm(sample_memory),
            memory_mask=sample_memory_mask)

        # Compute feature-feature interactions and reshape result accordingly.
        interactions = interactions.reshape(interactions.shape[1], -1, self.__feature_size)
        feature_memory = feature_memory.reshape(interactions.shape)
        interactions = self.feature_feature_associations(
            self.feature_norm(interactions),
            memory=self.feature_norm(feature_memory),
            memory_mask=feature_memory_mask
        ).reshape(*input.shape)

        return interactions


class Hopular(LightningModule):
    """
    Implementation of Hopular: Modern Hopfield Networks for Tabular Data.
    """
    PerformanceResult = namedtuple(
        typename=r'PerformanceResult',
        field_names=[
            r'loss_feature', r'loss_target',
            r'accuracy_feature', r'accuracy_target',
            r'feature_count', r'target_count'
        ]
    )

    class TrainingPhase(Enum):
        """
        Enumeration of available training phases used during the training of Hopular.
        """
        RESET = 0x0
        WARMUP = 0x1
        COOLDOWN = 0x2

    def __init__(self,

                 # Hopular-specific arguments.
                 input_sizes: List[int],
                 target_discrete: List[int],
                 target_numeric: List[int],
                 feature_discrete: List[int],
                 memory: torch.Tensor,
                 memory_ids: Optional[torch.Tensor] = None,
                 feature_size: int = 32,
                 hidden_size: int = 32,
                 hidden_size_factor: float = 1.0,
                 num_heads: int = 8,
                 scaling_factor: float = 1.0,
                 input_dropout: float = 0.1,
                 lookup_dropout: float = 0.1,
                 output_dropout: float = 0.1,
                 memory_ratio: float = 1.0,
                 num_blocks: int = 4,

                 # Optimizer-specific arguments.
                 initial_feature_loss_weight: float = 1.0,
                 final_feature_loss_weight: float = 0.0,
                 learning_rate: float = 1e-3,
                 gamma: float = 1.0,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 0.0,
                 lookup_steps: int = 5,
                 lookup_ratio: float = 0.8,
                 warmup_ratio: float = 0.5,
                 num_steps_per_cycle: int = 1,
                 cold_restart: bool = False,
                 asynchronous_weights: bool = False,

                 # Dataset-specific arguments.
                 feature_mean: Optional[torch.Tensor] = None,
                 feature_stdv: Optional[torch.Tensor] = None):
        """
        Initialize Hopular.

        :param input_sizes: original feature sizes (class count for discrete features)
        :param target_discrete: indices of discrete targets
        :param target_numeric: indices of continuous targets
        :param feature_discrete: indices of discrete features
        :param memory: external memory populated with the complete training set
        :param memory_ids: original sample indices used for prohibiting specific memory lookups
        :param feature_size: size of the embedding space of a single feature
        :param hidden_size: size of the Hopfield association space
        :param hidden_size_factor: factor for scaling the size of the Hopfield association space
        :param num_heads: count of modern Hopfield networks
        :param scaling_factor: factor for scaling beta to steer the retrieval type
        :param input_dropout: probability of embedding dropout regularization
        :param lookup_dropout: probability of Hopfield lookup dropout regularization
        :param output_dropout: probability of summarization dropout regularization
        :param memory_ratio: sub-sample ratio of external memory during training
        :param num_blocks: count of Hopular blocks, or count of iterative refinement steps
        :param initial_feature_loss_weight: initial weighting factor of self-supervised loss
        :param final_feature_loss_weight: final weighting factor of self-supervised loss
        :param learning_rate: base step size to be used for parameter updates
        :param gamma: decaying factor of learning rate and initial feature loss weight w.r.t. training cycles
        :param betas: coefficients for the running averages computed in the LAMB optimizer
        :param weight_decay: L2 decaying factor of model parameters used during training
        :param lookup_steps: count of fast weight updates before the slow weight update takes place
        :param lookup_ratio: ratio between fast and slow weights steering the slow weight updates
        :param warmup_ratio: count of steps before learning rate and feature loss weight annealing starts
        :param num_steps_per_cycle: count of steps of a single training cycle
        :param cold_restart: apply warmup and do not apply gamma decay at the beginning of a new training cycle
        :param asynchronous_weights: do not synchronize fast and slow weights
        :param feature_mean: feature means used for feature shifting
        :param feature_stdv: feature standard deviations used for feature scaling
        """
        betas = list(beta for beta in betas)
        super(Hopular, self).__init__()
        self.__input_sizes = input_sizes
        self.__feature_size = feature_size
        self.__hidden_size = hidden_size
        self.__hidden_size_factor = hidden_size_factor
        self.__num_heads = num_heads
        self.__target_discrete = target_discrete
        self.__target_numeric = target_numeric
        self.__feature_discrete = feature_discrete
        self.__scaling_factor = scaling_factor
        self.__input_dropout = input_dropout
        self.__lookup_dropout = lookup_dropout
        self.__output_dropout = output_dropout
        self.__memory_ratio = memory_ratio
        self.__num_blocks = num_blocks
        self.__initial_feature_loss_weight = initial_feature_loss_weight
        self.__final_feature_loss_weight = final_feature_loss_weight
        self.__learning_rate = learning_rate
        self.__gamma = gamma
        self.__betas = betas
        self.__weight_decay = weight_decay
        self.__lookup_steps = lookup_steps
        self.__lookup_ratio = lookup_ratio
        self.__warmup_ratio = warmup_ratio
        self.__cold_restart = cold_restart
        self.__asynchronous_weights = asynchronous_weights
        self.__feature_mean = feature_mean
        self.__feature_stdv = feature_stdv
        assert all((input_size >= 1 for input_size in self.__input_sizes)), r'Invalid input sizes specified!'
        assert 0.0 <= initial_feature_loss_weight <= 1.0, r'Invalid initial feature loss weight!'
        assert 0.0 <= final_feature_loss_weight <= 1.0, r'Invalid final feature loss weight!'
        assert 0.0 < self.__memory_ratio <= 1.0, r'Invalid memory ratio!'
        self.__feature_boundaries = torch.cumsum(torch.as_tensor([0] + self.__input_sizes), dim=0)
        self.__feature_boundaries = (self.__feature_boundaries[:-1], self.__feature_boundaries[1:])

        # Compute annealing thresholds and restart auxiliaries.
        self.__restarts = 0
        self.__cycle_step = 0
        self.__annealing_step = 1
        self.__annealing_offset = 0
        self.__warmup_steps = math.floor(self.__warmup_ratio * num_steps_per_cycle) - 1
        self.__cooldown_steps = num_steps_per_cycle - self.__warmup_steps
        assert self.__warmup_steps + self.__cooldown_steps == num_steps_per_cycle, r'Invalid training phases!'

        # Feature-(type/position)-specific embeddings.
        self.__input_size = self.__feature_size * len(self.__input_sizes)
        self.embeddings = EmbeddingBlock(
            input_sizes=[size + 1 for size in self.__input_sizes],
            feature_size=self.__feature_size,
            feature_discrete=self.__feature_discrete,
            dropout_probability=self.__input_dropout
        )

        # Hopular blocks including memory.
        self.register_buffer(name=r'memory', tensor=memory)
        self.register_buffer(name=r'memory_ids', tensor=torch.arange(len(memory)) if memory_ids is None else memory_ids)
        self.hopular_blocks = torch.nn.ModuleList(modules=[HopularBlock(
            input_size=self.__input_size,
            feature_size=self.__feature_size,
            hidden_size=self.__hidden_size,
            hidden_size_factor=self.__hidden_size_factor,
            num_heads=self.__num_heads,
            scaling_factor=self.__scaling_factor,
            dropout_probability=self.__lookup_dropout
        ) for _ in range(self.__num_blocks)])

        # Feature-specific summarizations.
        self.summarizations = SummarizationBlock(
            input_sizes=self.__input_sizes,
            feature_size=self.__feature_size,
            dropout_probability=self.__output_dropout
        )

        # Register loss functions and auxiliaries.
        self.annealing_factor = self.__initial_feature_loss_weight
        self.loss_numeric = torch.nn.MSELoss(reduction=r'mean')
        self.loss_discrete = torch.nn.CrossEntropyLoss(reduction=r'mean')

        # Register hyperparameters for logging.
        self.save_hyperparameters(ignore=[r'memory', r'memory_ids'])

    @classmethod
    def from_data_module(cls,
                         data_module: DataModule,
                         **kwargs: Dict[str, Any]) -> r'Hopular':
        """
        Initialize Hopular from a pre-instantiated data module.

        :param data_module: module encapsulating a dataset
        :param kwargs: additional keyword arguments used for initializing Hopular
        :return: new Hopular instance
        """
        data_module.setup(stage=r'memory')
        return cls(
            input_sizes=data_module.dataset.sizes,
            target_discrete=data_module.dataset.target_discrete.tolist(),
            target_numeric=data_module.dataset.target_numeric.tolist(),
            feature_discrete=data_module.dataset.feature_discrete.tolist(),
            memory=data_module.memory,
            memory_ids=data_module.dataset.split_train,
            feature_mean=data_module.dataset.feature_mean,
            feature_stdv=data_module.dataset.feature_stdv,
            **kwargs
        )

    def _get_training_phase(self) -> TrainingPhase:
        """
        Get the current training phase.

        :return: current training phase
        """
        threshold = self.__cycle_step % (self.__warmup_steps + self.__cooldown_steps)
        state = self.TrainingPhase.COOLDOWN
        if threshold == 0:
            state = self.TrainingPhase.RESET
        elif threshold <= self.__warmup_steps:
            state = self.TrainingPhase.WARMUP
        return state

    def _reset_scheduler_state(self) -> None:
        """
        Reset internal states of the training schedulers.

        :return: None
        """
        self.__restarts += 1
        self.__cycle_step = 0

        # Adapt schedulers.
        schedulers = self.lr_schedulers()
        for scheduler in schedulers if hasattr(schedulers, r'__iter__') else [schedulers]:
            if issubclass(type(scheduler), DelayedScheduler):
                scheduler.reset()
                if not self.__cold_restart:
                    self.__annealing_offset = self.__warmup_steps
                    self.__warmup_steps = 0
                    scheduler.first_step = 0
                    if hasattr(scheduler.scheduler, r'T_max'):
                        scheduler.scheduler.T_max = (self.__warmup_steps + self.__cooldown_steps) - 1

        # Adapt annealing thresholds.
        self.__annealing_step = 1
        self.__initial_feature_loss_weight = max(
            self.__initial_feature_loss_weight * self.__gamma ** self.__restarts,
            self.__final_feature_loss_weight
        )
        self.annealing_factor = self.__initial_feature_loss_weight

    def _load_optimizer_cache(self) -> None:
        """
        Backup fast weights and load slow weights.

        :return: None
        """
        optimizers = self.optimizers()
        for optimizer in optimizers if hasattr(optimizers, r'__iter__') else [optimizers]:
            if issubclass(type(optimizer), Lookahead):
                optimizer.backup_and_load_cache()

    def _restore_optimizer_backup(self) -> None:
        """
        Restore fast weights.

        :return: None
        """
        optimizers = self.optimizers()
        for optimizer in optimizers if hasattr(optimizers, r'__iter__') else [optimizers]:
            if issubclass(type(optimizer), Lookahead):
                optimizer.clear_and_load_backup()

    def _compute_performance(self,
                             result: torch.Tensor,
                             data_noise: torch.Tensor,
                             data_unmasked: torch.Tensor,
                             data_missing: torch.Tensor) -> PerformanceResult:
        """
        Compute performance measures.

        :param result: final prediction of Hopular
        :param data_noise: positions of masked entries
        :param data_unmasked: original input without masked entries
        :param data_missing: positions of missing entries
        :return: various performance measures (see <PerformanceResult>).
        """

        # Compute feature- and target-type-specific losses.
        feature_count = 0
        loss_feature = torch.zeros(1, device=result.device)
        accuracy_feature = torch.zeros(1, device=result.device)
        target_count = 0
        loss_target = torch.zeros(1, device=result.device)
        accuracy_target = torch.zeros(1, device=result.device)
        for feature_index, start, end in zip(range(len(self.__feature_boundaries[0])), *self.__feature_boundaries):
            data_valid = torch.logical_not(data_missing[:, feature_index])
            data_valid.logical_and_(data_noise[:, feature_index])
            if not data_valid.any():
                continue

            # Compute loss according to feature type.
            prediction = result[data_valid, start:end]
            target = data_unmasked[data_valid, start:end]
            if feature_index in self.__feature_discrete:
                loss = self.loss_discrete(prediction, target.argmax(dim=1))
                accuracy = (prediction.detach().argmax(dim=1) == target.detach().argmax(dim=1)).float().mean()
                if feature_index in self.__target_discrete:
                    loss_target = loss_target + loss
                    accuracy_target += accuracy
                    target_count += 1
                else:
                    loss_feature = loss_feature + loss
                    accuracy_feature += accuracy
                    feature_count += 1
            else:
                if not self.training:
                    if self.__feature_stdv is not None:
                        assert not np.isnan(self.__feature_stdv[feature_index])
                        prediction = prediction * self.__feature_stdv[feature_index]
                        target = target * self.__feature_stdv[feature_index]
                    if self.__feature_mean is not None:
                        assert not np.isnan(self.__feature_mean[feature_index])
                        prediction = prediction + self.__feature_mean[feature_index]
                        target = target + self.__feature_mean[feature_index]
                loss = self.loss_numeric(prediction, target)
                if feature_index in self.__target_numeric:
                    loss_target = loss_target + loss
                    target_count += 1
                else:
                    loss_feature = loss_feature + loss
                    feature_count += 1

        # Compute weighted metrics.
        if feature_count > 0:
            loss_feature = loss_feature / feature_count
            accuracy_feature = accuracy_feature / feature_count
        if target_count > 0:
            loss_target = loss_target / target_count
            accuracy_target = accuracy_target / target_count

        return self.PerformanceResult(
            loss_feature=loss_feature,
            loss_target=loss_target,
            accuracy_feature=accuracy_feature,
            accuracy_target=accuracy_target,
            feature_count=feature_count,
            target_count=target_count
        )

    def reset_parameters(self) -> None:
        """
        Reset parameters of Hopular.

        :return: None
        """

        # Initialize feature-(type/position)-specific embeddings.
        self.embeddings.reset_parameters()

        # Initialize Hopular blocks excluding memory.
        for module in self.hopular_blocks:
            module.reset_parameters()

        # Initialize feature-specific summarizations.
        self.summarizations.reset_parameters()

    def forward(self,
                input: torch.Tensor,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply Hopular.

        :param input: masked samples, masking positions, unmasked samples, missing positions and original sample indices
        :param memory_mask: mask for prohibiting specific sample lookups
        :return: refined final prediction
        """

        # Compute feature-(type/position)-specific embeddings as well as corresponding memory.
        embeddings = self.embeddings(input).unsqueeze(dim=0)
        embeddings_memory = self.embeddings(self.memory).unsqueeze(dim=0)

        # Optionally subsample memory.
        iteration_memory = embeddings_memory
        iteration_mask = memory_mask

        if self.training and self.__memory_ratio < 1.0:
            memory_indices = torch.randperm(embeddings_memory.shape[1], device=embeddings_memory.device)
            memory_indices = memory_indices[:max(1, int(self.__memory_ratio * memory_indices.shape[0]))]
            iteration_memory = iteration_memory[:, memory_indices]
            iteration_mask = iteration_mask[:, memory_indices] if iteration_mask is not None else None

        # Apply Hopular iterations.
        hopular_iteration = embeddings
        for hopular_block in self.hopular_blocks:
            hopular_iteration = hopular_block(
                hopular_iteration,
                sample_memory=iteration_memory, sample_memory_mask=iteration_mask,
                feature_memory=embeddings, feature_memory_mask=None
            )

        # Compute summarizations.
        hopular_iteration = hopular_iteration.reshape(hopular_iteration.shape[1], -1)
        return self.summarizations(hopular_iteration)

    def training_step(self,
                      batch: Tuple[torch.Tensor, ...],
                      batch_index: int) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step of Hopular.

        :param batch: masked samples, masking positions, unmasked samples, missing positions and original sample indices
        :param batch_index: index of the current mini-batch
        :return: combined loss of Hopular
        """
        data_masked, data_noise, data_unmasked, data_missing, data_indices = batch
        memory_mask = data_indices.view(-1, 1) == self.memory_ids.view(1, -1)
        result = self(data_masked, memory_mask=memory_mask)

        performance_result = self._compute_performance(
            result=result,
            data_noise=data_noise,
            data_unmasked=data_unmasked,
            data_missing=data_missing
        )

        # Compute weighted metrics.
        annealing_factor = self.annealing_factor
        if performance_result.feature_count <= 0:
            assert performance_result.target_count > 0, r'No element for computing loss found!'
            feature_loss, feature_accuracy = 0.0, 0.0
            target_loss, target_accuracy = performance_result.loss_target, performance_result.accuracy_target
            loss = target_loss
            annealing_factor = 0.0
        elif performance_result.target_count <= 0:
            assert performance_result.feature_count > 0, r'No element for computing loss found!'
            feature_loss, feature_accuracy = performance_result.loss_feature, performance_result.accuracy_feature
            target_loss, target_accuracy = 0.0, 0.0
            loss = feature_loss
            annealing_factor = 1.0
        else:
            feature_loss, feature_accuracy = performance_result.loss_feature, performance_result.accuracy_feature
            target_loss, target_accuracy = performance_result.loss_target, performance_result.accuracy_target
            loss = annealing_factor * feature_loss + (1.0 - annealing_factor) * target_loss

        # Log weighted metrics.
        self.log_dict({
            r'loss_feature/train': float(feature_loss), r'step': float(self.current_epoch)
        }, on_step=False, on_epoch=True)
        self.log_dict({
            r'loss/train': float(loss), r'step': float(self.current_epoch)
        }, on_step=False, on_epoch=True)
        self.log_dict({
            r'loss_target/train': float(target_loss), r'step': float(self.current_epoch)
        }, on_step=False, on_epoch=True)
        if len(np.setdiff1d(self.__feature_discrete, self.__target_discrete)) > 0:
            self.log_dict({
                r'accuracy_feature/train': float(feature_accuracy), r'step': float(self.current_epoch)
            }, on_step=False, on_epoch=True)
        if len(self.__target_discrete) > 0:
            self.log_dict({
                r'accuracy_target/train': float(target_accuracy), r'step': float(self.current_epoch)
            }, on_step=False, on_epoch=True)
        self.log_dict({
            r'annealing_factor': float(annealing_factor), r'step': float(self.current_epoch)
        }, on_step=False, on_epoch=True)
        self.log_dict({
            r'learning_rate': self.lr_schedulers().get_lr()[0], r'step': float(self.current_epoch)
        }, on_step=False, on_epoch=True)

        return {r'loss': loss}

    def validation_step(self,
                        batch: Tuple[torch.Tensor, ...],
                        batch_index: int) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step of Hopular.

        :param batch: masked samples, masking positions, unmasked samples, missing positions and original sample indices
        :param batch_index: index of the current mini-batch
        :return: combined loss of Hopular
        """
        data_masked, data_noise, data_unmasked, data_missing = batch
        result = self(data_masked, memory_mask=None)

        performance_result = self._compute_performance(
            result=result,
            data_noise=data_noise,
            data_unmasked=data_unmasked,
            data_missing=data_missing
        )

        # Compute weighted metrics.
        assert performance_result.target_count > 0, r'No element for computing loss found!'
        loss, accuracy = performance_result.loss_target, performance_result.accuracy_target

        # Log weighted metrics.
        self.log_dict({
            r'loss_target/val': float(loss), r'step': float(self.current_epoch)
        }, on_step=False, on_epoch=True)
        self.log_dict({
            r'loss/val': float(loss), r'step': float(self.current_epoch)
        }, on_step=False, on_epoch=True)
        should_log = self._get_training_phase() == self.TrainingPhase.COOLDOWN
        if len(self.__target_discrete) > 0:
            hp_metric = float(accuracy) if should_log else float(r'-inf')
            self.log_dict({
                r'accuracy_target/val': float(accuracy), r'step': float(self.current_epoch)
            }, on_step=False, on_epoch=True)
        else:
            hp_metric = float(loss) if should_log else float(r'+inf')
        self.log_dict({
            r'hp_metric/val': hp_metric, r'step': float(self.current_epoch)
        }, on_step=False, on_epoch=True)

        return {r'loss': loss}

    def test_step(self,
                  batch: Tuple[torch.Tensor, ...],
                  batch_index: int) -> Dict[str, torch.Tensor]:
        """
        Perform a single test step of Hopular.

        :param batch: masked samples, masking positions, unmasked samples, missing positions and original sample indices
        :param batch_index: index of the current mini-batch
        :return: combined loss of Hopular
        """
        data_masked, data_noise, data_unmasked, data_missing = batch
        result = self(data_masked, memory_mask=None)

        performance_result = self._compute_performance(
            result=result,
            data_noise=data_noise,
            data_unmasked=data_unmasked,
            data_missing=data_missing
        )

        # Compute weighted metrics.
        assert performance_result.target_count > 0, r'No element for computing loss found!'
        loss, accuracy = performance_result.loss_target, performance_result.accuracy_target

        # Log weighted metrics.
        self.log_dict({
            r'loss_target/test': float(loss), r'step': float(self.current_epoch)
        }, on_step=False, on_epoch=True)
        self.log_dict({
            r'loss/test': float(loss), r'step': float(self.current_epoch)
        }, on_step=False, on_epoch=True)
        if len(self.__target_discrete) > 0:
            self.log_dict({
                r'accuracy_target/test': float(accuracy), r'step': float(self.current_epoch)
            }, on_step=False, on_epoch=True)

        return {r'loss': loss}

    def on_train_epoch_end(self, **kwargs) -> None:
        """
        Adapt training cycle and scheduler annealing states.

        :param **kwargs: additional arguments available at the end of a single training epoch
        :return: None
        """
        # Adapt cycle step and optionally restart scheduler.
        self.__cycle_step += 1
        if self._get_training_phase() == self.TrainingPhase.RESET:
            self._reset_scheduler_state()

        # Adapt annealing factor.
        if self._get_training_phase() == self.TrainingPhase.COOLDOWN:
            feature_loss_weight_range = self.__initial_feature_loss_weight - self.__final_feature_loss_weight
            step_ratio = self.__annealing_step / (self.__cooldown_steps - 1)
            self.annealing_factor = self.__final_feature_loss_weight
            self.annealing_factor += 0.5 * feature_loss_weight_range * (1.0 + math.cos(step_ratio * math.pi))
            self.__annealing_step += 1

    def on_validation_start(self) -> None:
        """
        Load slow weights for validation.

        :return: None
        """
        self._load_optimizer_cache()

    def on_validation_end(self) -> None:
        """
        Restore fast weights for further training.

        :return: None
        """
        self._restore_optimizer_backup()

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        """
        Set up optimizers and schedulers using during the training of Hopular.

        :return: list of optimizers and list of schedulers
        """
        optimizer = Lookahead(
            optimizer=Lamb(
                params=self.parameters(),
                lr=self.__learning_rate,
                betas=self.__betas,
                weight_decay=self.__weight_decay
            ),
            la_steps=self.__lookup_steps,
            la_alpha=self.__lookup_ratio,
            pullback_momentum=r'none',
            synchronize_weights=self.__asynchronous_weights
        )
        learning_rate_scheduler = DelayedScheduler(
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=math.ceil((1.0 - self.__warmup_ratio) * (self.__warmup_steps + self.__cooldown_steps)),
                last_epoch=-1
            ),
            first_step=math.floor(self.__warmup_ratio * (self.__warmup_steps + self.__cooldown_steps)),
            gamma=self.__gamma
        )
        return [optimizer], [learning_rate_scheduler]
