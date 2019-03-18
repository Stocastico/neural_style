#!/usr/bin/env python3

import colorama
import argparse
import logging
from .logging_utils import logger
from .neural_style_basic import NeuralStyleGatys

# init colorama
colorama.init(autoreset=True)

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algorithm', default='basic', help='Neural style algorithm used')
parser.add_argument('-s', '--stile_img_path', required=True, help="Path to input style image")
parser.add_argument('-c', '--content_img_path', required=True, help="Path to input content image")
parser.add_argument('-o', '--output_img', required=False, default="output.png", help="Path of output image")
parser.add_argument('-i', '--iterations', required=False, default=10, help="Number of iterations", type=int)
parser.add_argument('--verbosity', help='specify log level (0 is minimum)', type=int, default=2, choices=[0, 1, 2, 3])
args = parser.parse_args()

if args.verbosity == 0:
    logger.setLevel(logging.ERROR)
elif args.verbosity == 1:
    logger.setLevel(logging.WARNING)
elif args.verbosity == 2:
    logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.DEBUG)

nst = NeuralStyleGatys(style_img_path=args.style_img_oath, content_img_path=args.content_img_path,
                       output_path=args.output_img, iterations=args.iterations)




# Close algo and reset logger
logger.info("Closing algorithm")

# close colorama
colorama.deinit()