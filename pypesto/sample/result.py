import numpy as np
from typing import Iterable


class McmcPtResult(dict):
    """The result of a sampler run using Markov-chain Monte Carlo, and
    optionally parallel tempering.

    Can be used like a dict.

    Parameters
    ----------
    trace_x: [n_chain, n_iter, n_par]
        Parameters.
    trace_neglogpost: [n_chain, n_iter]
        Negative log posterior values.
    trace_neglogprior: [n_chain, n_iter]
        Negative log prior values.
    betas: [n_chain]
        The associated inverse temperatures.
    burn_in: [n_chain]
        The burn in index.
    time: [n_chain]
        The computation time.
    auto_correlation: [n_chain]
        The estimated chain autocorrelation.
    effective_sample_size: [n_chain]
        The estimated effective sample size.
    message: str
        Textual comment on the profile result.
    debug: bool
        Whether to return additional information
        from the Markov chain.
    cum_chain_acceptance_rate: [n_chain, n_iter]
        The cumulative acceptance rate of the chains.
    cum_accepted_samples: [n_chain, n_iter]
        The cumulative accepted samples of the chains.
    accepted_swaps: [n_chain-1, n_iter]
        The number of accepted swaps between
        tempered chains.
    swap_acceptance_rate: [n_chain-1]
        The acceptance rate of swaps between
        tempered chains.
    covariance_scale_history: np.ndarray
        The scaling factor of the single-chain proposal
        covariance matrices, which is adapted to
        accomplish an overall 23% acceptance rate.
    covariance_history: np.ndarray
        Single-chain proposal covariance matrix.
    temperatures: [n_chain, n_iter]
        The temperatures of all tempered chains
        across all samples in the Markov chain.

    Here, `n_chain` denotes the number of chains, `n_iter` the number of
    iterations (i.e., the chain length), and `n_par` the number of parameters.
    """

    def __init__(self,
                 trace_x: np.ndarray,
                 trace_neglogpost: np.ndarray,
                 trace_neglogprior: np.ndarray,
                 betas: Iterable[float],
                 burn_in: int = None,
                 time: float = 0.,
                 auto_correlation: float = None,
                 effective_sample_size: float = None,
                 message: str = None,
                 debug: bool = False,
                 cum_chain_acceptance_rate: np.ndarray = None,
                 cum_accepted_samples: np.ndarray = None,
                 accepted_swaps: np.ndarray = None,
                 swap_acceptance_rate: np.ndarray = None,
                 covariance_scale_history: np.ndarray = None,
                 covariance_history: np.ndarray = None,
                 temperatures: np.ndarray = None):
        super().__init__()

        self.trace_x = trace_x
        self.trace_neglogpost = trace_neglogpost
        self.trace_neglogprior = trace_neglogprior
        self.betas = betas
        self.burn_in = burn_in
        self.time = time
        self.debug = debug
        self.auto_correlation = auto_correlation
        self.effective_sample_size = effective_sample_size
        self.message = message
        # additional for debugging
        self.cum_chain_acceptance_rate = cum_chain_acceptance_rate
        self.cum_accepted_samples = cum_accepted_samples
        self.accepted_swaps = accepted_swaps
        self.swap_acceptance_rate = swap_acceptance_rate
        self.covariance_scale_history = covariance_scale_history
        self.covariance_history = covariance_history
        self.temperatures = temperatures

        if trace_x.ndim != 3:
            raise ValueError(f"trace_x.ndim not as expected: {trace_x.ndim}")
        if trace_neglogpost.ndim != 2:
            raise ValueError("trace_neglogpost.ndim not as expected: "
                             f"{trace_neglogpost.ndim}")
        if trace_neglogprior.ndim != 2:
            raise ValueError("trace_neglogprior.ndim not as expected: "
                             f"{trace_neglogprior.ndim}")
        if trace_x.shape[0] != trace_neglogpost.shape[0] \
                or trace_x.shape[1] != trace_neglogpost.shape[1]:
            raise ValueError("Trace dimensions do not match:"
                             f"trace_x.shape={trace_x.shape},"
                             f"trace_neglogpost.shape={trace_neglogpost.shape}") # noqa
        if trace_x.shape[0] != trace_neglogprior.shape[0] \
                or trace_x.shape[1] != trace_neglogprior.shape[1]:
            raise ValueError("Trace dimensions do not match:"
                             f"trace_x.shape={trace_x.shape},"
                             f"trace_neglogprior.shape={trace_neglogprior.shape}") # noqa
        if trace_neglogpost.shape[0] != trace_neglogprior.shape[0] \
                or trace_neglogpost.shape[1] != trace_neglogprior.shape[1]:
            raise ValueError("Trace dimensions do not match:"
                             f"trace_neglogpost.shape={trace_neglogpost.shape}," # noqa
                             f"trace_neglogprior.shape={trace_neglogprior.shape}") # noqa

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
