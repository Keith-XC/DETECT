import ollama
import os
import base64


class OllamaVLMClient:
    """Ollama client wrapper for Vision Language Models (VLM) with Thinking support."""

    def __init__(self, host: str = None, model: str = "qwen3-vl:8b-thinking"):
        """
        Initialize the Ollama VLM client.

        Args:
            host: Ollama server URL.
            model: VLM model name (default: qwen3-vl:8b-thinking).
        """
        self.OLLAMA_HOST = host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.MODEL_NAME = model
        self.client = ollama.Client(host=self.OLLAMA_HOST)

    def _get_image_data(self, image_path: str):
        """Read image file and return bytes."""
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found.")
            return None
        with open(image_path, "rb") as f:
            return f.read()

    def run_vlm_prompt(self, prompt: str, image_path: str = None, temperature: float = 0):
        """
        Send a prompt with an optional image to the VLM.

        Args:
            prompt: The text instruction.
            image_path: Path to the local image file.
            temperature: Temperature for sampling, higher values produce more diverse outputs.
        """
        print(f"\n--- Processing with {self.MODEL_NAME} ---")

        images = []
        if image_path:
            img_data = self._get_image_data(image_path)
            if img_data:
                images.append(img_data)

        try:
            # Using 'generate' for single vision-text tasks
            response = self.client.generate(
                model=self.MODEL_NAME,
                prompt=prompt,
                images=images,
                options={
                'temperature': temperature,
                'top_k': 1,
                'top_p': 0.1
                }
            )
            return response['response']
        except Exception as e:
            print(f"Error: {e}")
            return None

    def start_vlm_chat(self):
        """Interactive chat session supporting text, images, and CoT thinking."""
        print(f"\n--- VLM Chat Session: {self.MODEL_NAME} ---")
        print("Commands: 'exit' to quit, 'image' to attach a picture, 'clear' to reset.")

        messages = []
        current_images = []

        while True:
            user_input = input("\nYou (Text): ").strip()

            if user_input.lower() == 'exit':
                break

            if user_input.lower() == 'clear':
                messages = []
                current_images = []
                print("History cleared.")
                continue

            if user_input.lower() == 'image':
                path = input("Enter image path: ").strip().replace("'", "").replace('"', "")
                img_data = self._get_image_data(path)
                if img_data:
                    current_images = [img_data]
                    print(f"Image '{path}' attached. Now ask something about it!")
                continue

            # Construct message with image if present
            msg_content = {'role': 'user', 'content': user_input}
            if current_images:
                msg_content['images'] = current_images

            messages.append(msg_content)

            print(f"\nAssistant is thinking/responding...")
            try:
                full_response = ""
                # Stream the response to see the <think> process in real-time
                for chunk in self.client.chat(model=self.MODEL_NAME, messages=messages, stream=True):
                    content = chunk['message']['content']
                    print(content, end="", flush=True)
                    full_response += content

                print("\n")
                messages.append({'role': 'assistant', 'content': full_response})
                current_images = []  # Clear image after one use to avoid repeating it in context unnecessarily
            except Exception as e:
                print(f"\nChat Error: {e}")


if __name__ == "__main__":
    # Ensure you have run: ollama run qwen3-vl:8b-thinking
    vlm = OllamaVLMClient()

    # Start interactive session
    try:
        vlm.start_vlm_chat()
    except KeyboardInterrupt:
        print("\nSession ended.")