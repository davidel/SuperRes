import argparse

import cv2
import utils as ut


def main(args):
  print(f'OpenCV: {cv2.__version__}')

  if args.scale_factor is not None:
    scale_spec = args.scale_factor
  else:
    scale_spec = (args.scale_size_x, args.scale_size_y)

  img = ut.super_scale(args.input_file, scale_spec,
                       model=args.model,
                       max_size=args.max_size)

  print(f'Saving output image to {args.output_file}')
  cv2.imwrite(args.output_file, img)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='UpScale using Super Resolution',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--input_file', type=str, required=True,
                      help='The path to the input image file')
  parser.add_argument('--output_file', type=str, required=True,
                      help='The path to the output image file')
  parser.add_argument('--scale_factor', type=float,
                      help='The uniform scale factor')
  parser.add_argument('--scale_size_x', type=int,
                      help='The size of the final image X (width)')
  parser.add_argument('--scale_size_y', type=int,
                      help='The size of the final image Y (height)')
  parser.add_argument('--model', type=str, default='edsr',
                      help='The DNN model to be used for scaling')
  parser.add_argument('--max_size', type=int, default=500,
                      help='The maximum tile size used during the upscaling process')

  args = parser.parse_args()
  main(args)

