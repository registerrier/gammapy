# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.config import ConfigNamespace, ConfigItem


class GammapyConfig(ConfigNamespace):
    """Gammapy configuration system"""

    rootname = "gammapy"

    # Logging
    logging_level = ConfigItem(
        ["info", "warning", "debug", "error", "critical"], "logging level"
    )

    mapaxis_label_template = ConfigItem(
        "{quantity} [{unit}]", "Label template for MapAxis in a plot"
    )
    mapaxis_unit_string_format = ConfigItem(
        "latex_inline", "String format to represent axis units in plot"
    )


# Instantiate the configuration so it can be imported
gammapy_config = GammapyConfig()
