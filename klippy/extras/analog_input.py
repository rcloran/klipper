# Support generic analog inputs
#
# Copyright (C) 2025  Russell Cloran <rcloran@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

SAMPLE_COUNT = 8  # Take 8 subsamples within the MCU
SAMPLE_TIME = 0.001  # with a 0.001s gap between each
REPORT_TIME = 1.000  # and report their average every 1s


class LimitHelper:
    def __init__(self, config, idx):
        main_key = "limit%d" % (idx, )
        over_key = "over_%s_gcode" % (main_key, )
        under_key = "under_%s_gcode" % (main_key, )

        printer = config.get_printer()
        self.limit = config.getfloat(main_key)
        self.gcode = printer.lookup_object('gcode')
        gcode_macro = printer.load_object(config, 'gcode_macro')
        self.over_template = gcode_macro.load_template(config, over_key, '')
        self.under_template = gcode_macro.load_template(config, under_key, '')

    def callback(self, read_time, value, last_value):
        if value < self.limit and last_value >= self.limit:
            self.under_template.run_gcode_from_command()
        elif value > self.limit and last_value <= self.limit:
            self.over_template.run_gcode_from_command()

class PrinterAnalogInput:
    def __init__(self, config):
        self.name = config.get_name()
        self.unit = config.get("unit", "")
        self.scale = config.getfloat("scale", 1.0)
        self.offset = config.getfloat("offset", 0.0)
        self._limit_helpers = []
        for i in range(1, 1000):
            if config.get("limit%d" % (i, ), None) is None:
                break
            self._limit_helpers.append(LimitHelper(config, i))

        self.last_value = 0.0

        printer = config.get_printer()
        ppins = printer.lookup_object("pins")
        self.mcu_adc = ppins.setup_pin("adc", config.get("sensor_pin"))
        self.mcu_adc.setup_adc_callback(REPORT_TIME, self.adc_callback)
        self.mcu_adc.setup_adc_sample(SAMPLE_TIME, SAMPLE_COUNT)
        query_adc = printer.load_object(config, "query_adc")
        query_adc.register_adc(self.name, self.mcu_adc)

    def adc_callback(self, read_time, read_value):
        value = read_value * self.scale + self.offset
        for helper in self._limit_helpers:
            helper.callback(read_time, value, self.last_value)
        self.last_value = value

    def stats(self, eventtime):
        return False, "%s: value=%.1f%s" % (self.name, self.last_value, self.unit)

    def get_status(self, eventtime):
        return {
            "value": round(self.last_value, 2),
            "unit": self.unit,
        }


def load_config_prefix(config):
    return PrinterAnalogInput(config)
