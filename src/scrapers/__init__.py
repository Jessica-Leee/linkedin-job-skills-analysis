"""Bundled collection helpers; external browser dependencies load on demand."""

from importlib import import_module

__version__ = "2.11.5"
_EXPORTS = {
    "Person": ("person", "Person"),
    "Institution": ("objects", "Institution"),
    "Experience": ("objects", "Experience"),
    "Education": ("objects", "Education"),
    "Contact": ("objects", "Contact"),
    "Company": ("company", "Company"),
    "Job": ("jobs", "Job"),
    "JobSearch": ("job_search", "JobSearch"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value
