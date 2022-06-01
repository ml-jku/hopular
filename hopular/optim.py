import math
import torch

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


class DelayedScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Wrapper for the delayed activation of a scheduler.
    """

    def __init__(self,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 first_step: int,
                 gamma: float):
        """
        Initialize delayed scheduler wrapper.

        :param scheduler: scheduler instance to wrap
        :param first_step: step when to activate the wrapped scheduler
        :param gamma: decaying factor of learning rate and initial feature loss weight w.r.t. training cycles
        """
        self.scheduler = scheduler
        self.first_step = first_step
        self.gamma = gamma
        self.cached_state = {}
        self.current_step = 0
        self.cached_state = {r'self': self.state_dict(), r'scheduler': self.scheduler.state_dict()}
        self.cached_state[r'self'][r'restarts'] = 1
        super(DelayedScheduler, self).__init__(scheduler.optimizer, scheduler.last_epoch, scheduler.verbose)

    def state_dict(self) -> Dict[str, Any]:
        """
        Get the current state of the scheduler wrapper.

        :return: state of scheduler wrapper
        """
        return {key: value for key, value in self.__dict__.items() if key not in (r'scheduler', r'cached_state')}

    def reset(self) -> None:
        """
        Reset the state of the scheduler wrapper and the wrapped scheduler.

        :return: None
        """

        # Adapt state and restart schedulers.
        self.cached_state[r'scheduler'][r'base_lrs'] = [
            lr * self.gamma ** self.cached_state[r'self'][r'restarts'] for
            lr in self.cached_state[r'scheduler'][r'base_lrs']]
        self.cached_state[r'scheduler'][r'_last_lr'] = self.cached_state[r'scheduler'][r'base_lrs']
        self.load_state_dict(state_dict=self.cached_state[r'self'])
        self.scheduler.load_state_dict(state_dict=self.cached_state[r'scheduler'])

        # Adapt state for next restart.
        self.cached_state[r'self'][r'restarts'] += 1

    def get_lr(self) -> float:
        """
        Get the current learning rate of the wrapper scheduler.

        :return: current learning rate
        """
        return self.scheduler.get_last_lr()

    def step(self) -> None:
        """
        Perform a single step of the scheduler wrapper and the wrapped scheduler.

        :return: None
        """
        super().step()
        if self.current_step >= self.first_step:
            self.scheduler.step()
        self.current_step += 1


class Lookahead(torch.optim.Optimizer):
    """
    Lookahead optimizer wrapper [1]. Implementation is based on [2].
    Added capability of de-coupling fast and slow weights.

    [1] https://arxiv.org/abs/1907.08610
    [2] https://github.com/michaelrzhang/lookahead/blob/c4fd85f73cdaa6450eacf7d52d9050d9ff98a6e1/lookahead_pytorch.py
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 la_steps: int = 5,
                 la_alpha: float = 0.8,
                 pullback_momentum: str = r'none',
                 synchronize_weights: bool = True):
        """
        Initialize Lookahead optimizer wrapper.

        :param optimizer: optimizer instance to wrap
        :param la_steps: count of fast weight updates before the slow weight update takes place
        :param la_alpha: ratio between fast and slow weights steering the slow weight updates
        :param pullback_momentum: pull back momentum terms during lookahead step
        :param synchronize_weights: synchronize fast and slow weights
        """
        self.optimizer = optimizer
        self.alpha = la_alpha
        self.total_steps = la_steps
        self.pullback_momentum = pullback_momentum.lower()
        self.synchronize_weights = synchronize_weights
        assert self.pullback_momentum in [r'reset', r'pullback', r'none']
        defaults = dict(
            la_steps=la_steps,
            la_alpha=la_alpha,
            pullback_momentum=pullback_momentum,
            synchronize_weights=synchronize_weights)
        super(Lookahead, self).__init__(self.optimizer.param_groups, defaults)

        # Cache the current optimizer parameters.
        for group in self.optimizer.param_groups:
            for p in group[r'params']:
                state = self.state[p]
                state[r'cached_params'] = torch.zeros_like(p.data)
                state[r'cached_params'].copy_(p.data)
                if self.pullback_momentum == r'pullback':
                    state[r'cached_momentum'] = torch.zeros_like(p.data)

    def backup_and_load_cache(self) -> None:
        """
        Backup fast weights and load slow weights.

        :return: None
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def clear_and_load_backup(self) -> None:
        """
        Restore fast weights.

        :return: None
        """
        for group in self.optimizer.param_groups:
            for p in group[r'params']:
                param_state = self.state[p]
                p.data.copy_(param_state[r'backup_params'])
                del param_state[r'backup_params']

    def step(self,
             closure: Optional[Callable[[], Any]] = None) -> torch.Tensor:
        """
        Perform a single step of the Lookahead optimizer wrapper and the wrapped optimizer.

        :param closure: closure to be executed by the wrapped optimizer
        :return: loss as computed by the specified closure
        """
        loss = self.optimizer.step(closure)
        for group in self.optimizer.param_groups:
            for p in group[r'params']:

                # Initialize state.
                state = self.state[p]
                if r'step' not in state:
                    state[r'step'] = 0

                # Adapt and check internal state.
                state[r'step'] += 1
                if state[r'step'] >= self.total_steps:
                    state[r'step'] = 0

                    # Lookahead and cache the current optimizer parameters.
                    state[r'cached_params'].add_(p.data - state[r'cached_params'], alpha=self.alpha)
                    if self.synchronize_weights:
                        p.data.copy_(state[r'cached_params'])

                        if self.pullback_momentum == r'pullback':
                            internal_momentum = self.optimizer.state[p][r'momentum_buffer']
                            self.optimizer.state[p][r'momentum_buffer'] = internal_momentum.mul_(self.alpha).add_(
                                1.0 - self.alpha, param_state[r'cached_momentum'])
                            param_state[r'cached_momentum'] = self.optimizer.state[p][r'momentum_buffer']
                        elif self.pullback_momentum == r'reset':
                            if r'momentum_buffer' in self.optimizer.state[p]:
                                self.optimizer.state[p][r'momentum_buffer'] = torch.zeros_like(p.data)
                            else:
                                self.optimizer.state[p] = {}

        return loss


class Lamb(torch.optim.Optimizer):
    """
    LAMB optimizer [1]. Implementation based on [2].
    Re-introduced bias correction.

    [1] https://arxiv.org/abs/1904.00962
    [2] https://github.com/cybertronai/pytorch-lamb/blob/d3ab8dccf6717977c1ad0d6b95499f6b25bba41b/pytorch_lamb/lamb.py
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-6,
                 weight_decay: float = 0.0):
        """
        Initialize LAMB optimizer.

        :param params:
        :param lr: base step size to be used for parameter updates
        :param betas: coefficients for the running averages
        :param eps: additive term for numerical stability
        :param weight_decay: L2 decaying factor of model parameters
        """
        if lr <= 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if eps <= 0.0:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Lamb, self).__init__(params, defaults)

    def step(self,
             closure: Optional[Callable[[], Any]] = None) -> torch.Tensor:
        """
        Perform a single step of the LAMB optimizer.

        :param closure: closure to be executed to compute the current loss
        :return: loss as computed by the specified closure
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group[r'params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(r'Lamb does not support sparse gradients, consider SparseAdam instead.')

                # State initialization.
                state = self.state[p]
                if len(state) == 0:
                    assert all((element not in state for element in (r'state', r'exp_avg', r'exp_avg_sq')))
                    state[r'step'] = 0
                    state[r'exp_avg'] = torch.zeros_like(p.data)
                    state[r'exp_avg_sq'] = torch.zeros_like(p.data)

                # Extract current state elements.
                exp_avg, exp_avg_sq = state[r'exp_avg'], state[r'exp_avg_sq']
                beta1, beta2 = group[r'betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficients (<m_t> and <v_t>).
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Apply bias correction.
                bias_correction1 = 1.0 - beta1 ** state[r'step']
                bias_correction2 = 1.0 - beta2 ** state[r'step']
                step_size = group[r'lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Update optimizer state.
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0.0, 10.0)
                adam_step = exp_avg / exp_avg_sq.sqrt().add(group[r'eps'])
                if group[r'weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group[r'weight_decay'])

                adam_norm = adam_step.pow(2).sum().sqrt()
                trust_ratio = 1.0 if ((weight_norm == 0) or (adam_norm == 0)) else (weight_norm / adam_norm)
                state[r'weight_norm'] = weight_norm
                state[r'adam_norm'] = adam_norm
                state[r'trust_ratio'] = trust_ratio
                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss
