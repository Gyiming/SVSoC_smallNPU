from __future__ import with_statement
from __future__ import print_function
import os
import configparser
import random
import numpy as np 
import schedule
import bestE


def main():

    conf = configparser.ConfigParser()
    conf.read("soc_configure.cfg")
    sections = conf.sections()  
    a=conf.get("info","CPU1_enable")
    print(a)
    conf.set("info", "CPU1_enable", "10")
    a=conf.get("info","CPU1_enable")
    print(a)


if __name__ == '__main__':
	main()