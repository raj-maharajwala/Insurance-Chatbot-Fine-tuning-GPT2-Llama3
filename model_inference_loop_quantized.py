import os
import time
from pathlib import Path
from llama_cpp import Llama
from rich.console import Console
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class ModelConfig:
    # Optimized parameters for coherent responses and efficient performance on devices like MacBook Air M2
    model_name: str = "Raj-Maharajwala/Open-Insurance-LLM-Llama3-8B-GGUF"
    model_file: str = "open-insurance-llm-q4_k_m.gguf"
    # model_file: str = "open-insurance-llm-q8_0.gguf"  # 8-bit quantization; higher precision, better quality, increased resource usage
    # model_file: str = "open-insurance-llm-q5_k_m.gguf"  # 5-bit quantization; balance between performance and resource efficiency
    max_tokens: int = 1000  # Maximum number of tokens to generate in a single output
    temperature: float = 0.1  # Controls randomness in output; lower values produce more coherent responses (performs scaling distribution)
    top_k: int = 15  # After temperature scaling, Consider the top 15 most probable tokens during sampling
    top_p: float = 0.2  # After reducing the set to 15 tokens, Uses nucleus sampling to select tokens with a cumulative probability of 20%
    repeat_penalty: float = 1.2  # Penalize repeated tokens to reduce redundancy
    num_beams: int = 4  # Number of beams for beam search; higher values improve quality at the cost of speed
    n_gpu_layers: int = -2  # Number of layers to offload to GPU; -1 for full GPU utilization, -2 for automatic configuration
    n_ctx: int = 2048  # Context window size; Llama 3 models support up to 8192 tokens context length
    n_batch: int = 256  # Number of tokens to process simultaneously; adjust based on available hardware (suggested 512)
    verbose: bool = False  # True for enabling verbose logging for debugging purposes
    use_mmap: bool = False  # Memory-map model to reduce RAM usage; set to True if running on limited memory systems
    use_mlock: bool = True  # Lock model into RAM to prevent swapping; improves performance on systems with sufficient RAM
    offload_kqv: bool = True  # Offload key, query, value matrices to GPU to accelerate inference



class InsuranceLLM:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.llm_ctx = None
        self.console = Console()
        self.conversation_history: List[Dict[str, str]] = []
        
        self.system_message = (
            "This is a chat between a user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. "
            "The assistant should also indicate when the answer cannot be found in the context. "
            "You are an expert from the Insurance domain with extensive insurance knowledge and "
            "professional writer skills, especially about insurance policies. "
            "Your name is OpenInsuranceLLM, and you were developed by Raj Maharajwala. "
            "You are willing to help answer the user's query with a detailed explanation. "
            "In your explanation, leverage your deep insurance expertise, such as relevant insurance policies, "
            "complex coverage plans, or other pertinent insurance concepts. Use precise insurance terminology while "
            "still aiming to make the explanation clear and accessible to a general audience."
        )

    def download_model(self) -> str:
        try:
            with self.console.status("[bold green]Downloading model..."):
                model_path = hf_hub_download(
                    self.config.model_name,
                    filename=self.config.model_file,
                    local_dir=os.path.join(os.getcwd(), 'gguf_dir')
                )
            return model_path
        except Exception as e:
            self.console.print(f"[red]Error downloading model: {str(e)}[/red]")
            raise

    def load_model(self) -> None:
        try:
            quantized_path = os.path.join(os.getcwd(), "gguf_dir")
            directory = Path(quantized_path)

            try:
                model_path = str(list(directory.glob(self.config.model_file))[0])
            except IndexError:
                model_path = self.download_model()

            with self.console.status("[bold green]Loading model..."):
                self.llm_ctx = Llama(
                    model_path=model_path,
                    n_gpu_layers=self.config.n_gpu_layers,
                    n_ctx=self.config.n_ctx,
                    n_batch=self.config.n_batch,
                    num_beams=self.config.num_beams,
                    verbose=self.config.verbose,
                    use_mlock=self.config.use_mlock,
                    use_mmap=self.config.use_mmap,
                    offload_kqv=self.config.offload_kqv
                )
        except Exception as e:
            self.console.print(f"[red]Error loading model: {str(e)}[/red]")
            raise

    def build_conversation_prompt(self, new_question: str, context: str = "") -> str:
        prompt = f"System: {self.system_message}\n\n"
        
        # Add conversation history
        for exchange in self.conversation_history:
            prompt += f"User: {exchange['user']}\n\n"
            prompt += f"Assistant: {exchange['assistant']}\n\n"
        
        # Add the new question
        if context:
            prompt += f"User: Context: {context}\nQuestion: {new_question}\n\n"
        else:
            prompt += f"User: {new_question}\n\n"
            
        prompt += "Assistant:"
        return prompt

    def generate_response(self, prompt: str) -> Tuple[str, int, float]:
        if not self.llm_ctx:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.console.print("[bold cyan]Assistant: [/bold cyan]", end="")
        complete_response = ""
        token_count = 0
        start_time = time.time()

        try:
            for chunk in self.llm_ctx.create_completion(
                prompt,
                max_tokens=self.config.max_tokens,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                repeat_penalty=self.config.repeat_penalty,
                stream=True
            ):
                text_chunk = chunk["choices"][0]["text"]
                complete_response += text_chunk
                token_count += 1
                print(text_chunk, end="", flush=True)
            
            elapsed_time = time.time() - start_time
            print()
            return complete_response, token_count, elapsed_time
        except Exception as e:
            self.console.print(f"\n[red]Error generating response: {str(e)}[/red]")
            return f"I encountered an error while generating a response. Please try again or ask a different question.", 0, 0

    def run_chat(self):
        try:
            self.load_model()
            self.console.print("\n[bold green]Welcome to Open-Insurance-LLM![/bold green]")
            self.console.print("Enter your questions (type '/bye', 'exit', or 'quit' to end the session)\n")
            self.console.print("Optional: You can provide context by typing 'context:' followed by your context, then 'question:' followed by your question\n")
            self.console.print("Your conversation history will be maintained for context-aware responses.\n")
            
            total_tokens = 0
            
            while True:
                try:
                    user_input = self.console.input("[bold cyan]User:[/bold cyan] ").strip()

                    if user_input.lower() in ["exit", "/bye", "quit"]:
                        self.console.print(f"\n[dim]Total tokens: {total_tokens}[/dim]")
                        self.console.print("\n[bold green]Thank you for using OpenInsuranceLLM![/bold green]")
                        break

                    # Reset conversation with command
                    if user_input.lower() == "/reset":
                        self.conversation_history = []
                        self.console.print("[yellow]Conversation history has been reset.[/yellow]")
                        continue

                    context = ""
                    question = user_input
                    if "context:" in user_input.lower() and "question:" in user_input.lower():
                        parts = user_input.split("question:", 1)
                        context = parts[0].replace("context:", "").strip()
                        question = parts[1].strip()

                    prompt = self.build_conversation_prompt(question, context)
                    response, tokens, elapsed_time = self.generate_response(prompt)
                    
                    # Add to conversation history
                    self.conversation_history.append({
                        "user": question,
                        "assistant": response
                    })
                    
                    # Update total tokens
                    total_tokens += tokens
                    
                    # Print metrics
                    tokens_per_sec = tokens / elapsed_time if elapsed_time > 0 else 0
                    self.console.print(
                        f"[dim]Tokens: {tokens} || " +
                        f"Time: {elapsed_time:.2f}s || " +
                        f"Speed: {tokens_per_sec:.2f} tokens/sec[/dim]"
                    )
                    print()  # Add a blank line after each response
                    
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Input interrupted. Type '/bye', 'exit', or 'quit' to quit.[/yellow]")
                    continue
                except Exception as e:
                    self.console.print(f"\n[red]Error processing input: {str(e)}[/red]")
                    continue
        except Exception as e:
            self.console.print(f"\n[red]Fatal error: {str(e)}[/red]")
        finally:
            if self.llm_ctx:
                del self.llm_ctx


def main():
    try:
        config = ModelConfig()
        llm = InsuranceLLM(config)
        llm.run_chat()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nApplication error: {str(e)}")


if __name__ == "__main__":
    main()
