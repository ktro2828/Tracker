#!/usr/bin/env python


import argparse


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--target', type=str,
                        default=None, help='target video file path')
