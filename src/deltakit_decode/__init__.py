# (c) Copyright Riverlane 2020-2025.
"""Description of ``deltakit.decode`` namespace here."""

import importlib.metadata

from deltakit_decode._mwpm_decoder import PyMatchingDecoder

# It looks like these were intended to be separate, public modules.
# For now, import them as such. This can be reconsidered during API review.
from . import analysis, noise_sources, utils

__version__ = importlib.metadata.version(__package__)

# Prevent import of `importlib` (and any other non-public objects) from this module.
del importlib

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]
