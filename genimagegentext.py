import argparse
import os
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part

def generate_image(
    project_id: str, location: str, output_file: str, prompt: str
) -> vertexai.preview.vision_models.ImageGenerationResponse:
    """Generate an image using a text prompt.
    Args:
      project_id: Google Cloud project ID, used to initialize Vertex AI.
      location: Google Cloud region, used to initialize Vertex AI.
      output_file: Local path to the output image file.
      prompt: The text prompt describing what you want to see."""

    vertexai.init(project=project_id, location=location)

    model = ImageGenerationModel.from_pretrained("imagegeneration@002")

    images = model.generate_images(
        prompt=prompt,
        # Optional parameters
        number_of_images=1,
        seed=1,
        add_watermark=False,
    )

    images[0].save(location=output_file)

    return images


def generate_text(project_id: str, location: str) -> str:
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    # Load the model
    multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    # Get the current directory and set the image path
    image_path = os.path.join(os.getcwd(), "image.jpeg")
    # Read image file as binary data
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()
    # Query the model
    response = multimodal_model.generate_content(
        [
            # Add an example image
            Part.from_data(data=image_data, mime_type="image/jpeg"),
            # Add an example query
            "generate birthday wishes based on the image",
        ]
    )

    return response.text

generate_image(
    project_id='qwiklabs-gcp-00-000000000000',
    location='us-west1',
    output_file='image.jpeg',
    prompt='Create an image containing a bouquet of 2 sunflowers and 3 roses',
    )

response = generate_text('qwiklabs-gcp-00-000000000000', 'us-west1')
print(response)








# --------  Important: Variable declaration  --------

project_id = "project-id"
location = "REGION"

#  --------   Call the Function  --------

response = generate_text(project_id, location)
print(response)
