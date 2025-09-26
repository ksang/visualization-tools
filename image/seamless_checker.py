import cv2
import os
import numpy as np
import typer
from typing_extensions import Annotated

def seamless_checker(
    image_path: Annotated[
        str, typer.Argument(help="Path to the input image (PNG or JPG)")
    ],
    output_video_path: Annotated[
        str, typer.Argument(help="Path to save the output video")
    ] = "output/output.mp4",
    fps: Annotated[
        int, typer.Option(help="Frames per second for the output video")
    ] = 30,
    offset_step_percentage: Annotated[
        float,
        typer.Option(
            help="The percentage step for each offset increment (e.g., 1 for 1%)."
        ),
    ] = 1.0,
):
    """
    Creates a video demonstrating the seamlessness of an image by iteratively
    offsetting it in both x and y directions simultaneously based on percentage.
    """
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            typer.echo(f"Error: Could not read image from {image_path}")
            raise typer.Exit(code=1)

        height, width, _ = image.shape
        max_offset_x = width // 2
        max_offset_y = height // 2

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Calculate the number of frames based on the offset percentage
        num_frames = int(100 / offset_step_percentage) + 1
        
        # Iteratively offset the image and write frames to the video
        for i in range(num_frames):
            offset_percent = i * offset_step_percentage
            # Offset to top-right means moving the content of the image down and left
            x_offset_pixels = int(max_offset_x * (offset_percent / 100))
            y_offset_pixels = int(max_offset_y * (offset_percent / 100))

            # Create the offset image using tiling and slicing
            tiled_image = np.tile(image, (3, 3, 1))
            # The slice starts from the top-left of the center tile and moves
            # to the bottom-right as the offset increases.
            seamless_frame = tiled_image[
                height - y_offset_pixels : 2 * height - y_offset_pixels,
                width + x_offset_pixels : 2 * width + x_offset_pixels,
            ]
            out.write(seamless_frame)

        # Release everything when done
        out.release()
        typer.echo(f"Video saved successfully to {output_video_path}")

    except Exception as e:
        typer.echo(f"An error occurred: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(seamless_checker)