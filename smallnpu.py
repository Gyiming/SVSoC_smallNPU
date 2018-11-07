# -* - coding: UTF-8 -* -
from __future__ import with_statement
from __future__ import print_function
import os
import configparser
import random
import numpy as np 
import schedule
import bestE


def main():
    config=configparser.ConfigParser()
    enable_performance = [0 for i in range(11)]
    enable_energy = [0 for i in range(11)]
    start_time = [0 for i in range(10000)]
    sensing_time = [0 for i in range(10000)]
    ISP_time = [0 for i in range(10000)]