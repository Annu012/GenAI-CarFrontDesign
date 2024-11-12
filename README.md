
# GenAI-CarFrontDesign

## Overview

With advancements in artificial intelligence, big data, and cloud computing, generative AI applications have started revolutionizing design processes, enhancing productivity and efficiency over traditional CAD (Computer-Aided Design) tools. This project introduces a new methodology leveraging generative AI for designing car frontal forms. This approach is intended to streamline workflows for industrial designers, improve design productivity, and encourage a reevaluation of conventional design techniques.

## Methodology

The generative AI-based design methodology follows these main steps:

1. **Imagery Adjective Generation**: 
   - Imagery adjectives describing car frontal forms are generated through text-based prompts in the ChatGPT AI model.

2. **Image Generation**: 
   - Typical imagery adjectives are used as prompts in Midjourney, generating diverse car frontal forms aligned with these adjectives, forming a comprehensive reference form database.

3. **Base Form Selection & Target Imagery Definition**:
   - Select a base form and define target imageries.
   - Select reference forms from the database that match the target imagery for design inspiration.

4. **Form Blending with Bézier Curves**:
   - Using cubic Bézier curves, delineate main form elements from both base and reference forms.
   - Apply a form curve blending algorithm to produce a set of alternative designs.

5. **3D Rendering**:
   - Convert the alternatives into 3D renderings with Vega AI, providing realistic design visualization.

6. **Evaluation**:
   - Validate the alternatives through FAHP-based expert evaluations and consumer perceptual feedback.

## Key Features

- **Automated Image Generation**: Quickly generates various car frontal form concepts based on text prompts.
- **Flexible Design Adjustments**: Use form blending and Bézier curves for creating alternative designs.
- **Efficient 3D Rendering**: Convert 2D designs into 3D renderings with image-generative AIs like Vega AI.
- **User-Friendly Workflow**: No programming knowledge is required, making it accessible for designers from diverse backgrounds.

## Requirements

To run and adapt this methodology, you will need:

- Python 3.x
- Required AI libraries and platforms (ChatGPT, Midjourney, Vega AI)
- [NumPy](https://numpy.org/) for handling arrays and mathematical computations
- [FAHP Package](https://fahp.org/) for the evaluation phase (optional)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GenAI-CarFrontDesign.git
   ```
2. Install necessary libraries:
   ```bash
   pip install numpy fahp # Additional dependencies as required
   ```
3. Set up accounts for ChatGPT, Midjourney, and Vega AI if not already configured.

## Usage

1. **Generate Imagery Adjectives**:
   - Use ChatGPT to create descriptive adjectives for your desired car design.

2. **Generate Reference Images**:
   - Use Midjourney or another image-generative AI to produce car frontal forms based on generated adjectives.

3. **Create Alternatives**:
   - Apply the form curve blending algorithm on selected base and reference forms using cubic Bézier curves.

4. **Render 3D Images**:
   - Use Vega AI or a similar tool to render your alternatives as 3D images.

5. **Evaluate**:
   - Assess your designs using FAHP-based expert evaluation and consumer feedback.

## Example

An example design case study is included in the repository. This demonstrates each step, from generating imagery adjectives to obtaining consumer feedback on final design alternatives.

## Contributing

Contributions to enhance the methodology, add features, or improve workflow are welcome. Feel free to fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

This project is based on research into generative design, AI-assisted CAD, and industrial design workflows. Special thanks to developers of generative AI tools and platforms, including ChatGPT, Midjourney, and Vega AI, for their innovative contributions to the design industry.

---


