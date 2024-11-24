import os
import time
import logging
import sys
import traceback
import multiprocessing
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager
import psutil
from rich.console import Console
# from rich.progress import Progress, Progress, TimeElapsedColumn
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import datetime
from rich.logging import RichHandler
from rich.theme import Theme
from rich.traceback import install
# Install rich traceback handler
install(show_locals=True)

@dataclass
class ModelConfig:
    model_name: str = "Raj-Maharajwala/Open-Insurance-LLM-Llama3-8B-GGUF"
    model_file: str = "open-insurance-llm-q8_0.gguf"
    # model_file: str = "open-insurance-llm-q4_k_m.gguf"
    max_tokens: int = 1000
    top_k: int = 15
    top_p: float = 0.2
    repeat_penalty: float = 1.2
    num_beams: int = 4
    n_gpu_layers: int = -2 # -1 for complete GPU usage
    temperature: float = 0.1
    n_ctx: int = 2048 # 2048 - 8192 -> As per Llama 3 Full Capacity
    n_batch: int = 256
    verbose: bool = False
    use_mmap: bool = False
    use_mlock: bool = True
    offload_kqv: bool =True

class CustomFormatter(logging.Formatter):
    """Enhanced formatter with detailed context for different log levels"""  
    FORMATS = {
        logging.DEBUG: "🔍 %(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",
        logging.INFO: "ℹ️ %(asctime)s - %(name)s - [%(funcName)s] - %(levelname)s - %(message)s",
        logging.WARNING: "⚠️ %(asctime)s - %(name)s - [%(funcName)s] - %(levelname)s - %(message)s\nContext: %(pathname)s",
        logging.ERROR: "❌ %(asctime)s - %(name)s - [%(funcName)s:%(lineno)d] - %(levelname)s - %(message)s\nTraceback: %(exc_info)s",
        logging.CRITICAL: """🚨 %(asctime)s - %(name)s - %(levelname)s
Location: %(pathname)s:%(lineno)d
Function: %(funcName)s
Process: %(process)d
Thread: %(thread)d
Message: %(message)s
Memory: %(memory).2fMB
%(exc_info)s
"""
    }

    def format(self, record):
        # Add memory usage information
        if not hasattr(record, 'memory'):
            record.memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Add exception info for ERROR and CRITICAL
        if record.levelno >= logging.ERROR and not record.exc_info:
            record.exc_info = traceback.format_exc()
            
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        
        # Add performance metrics
        if hasattr(record, 'duration'):
            record.message = f"{record.message}\nDuration: {record.duration:.2f}s"
            
        return formatter.format(record)

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Enhanced logging setup with multiple handlers and log files"""
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (Path(log_dir) / f"l_{timestamp}")
    log_path.mkdir(exist_ok=True)
    # Create separate log files for different levels
    handlers = {
        'debug': logging.FileHandler(log_path / f"debug_{timestamp}.log"),
        'info': logging.FileHandler(log_path / f"info_{timestamp}.log"),
        'error': logging.FileHandler(log_path / f"error_{timestamp}.log"),
        'critical': logging.FileHandler(log_path / f"critical_{timestamp}.log"),
        'console': RichHandler(
            console=Console(theme=custom_theme),
            show_time=True,
            show_path=False,
            enable_link_path=True
        )
    }    
    # Set levels for handlers
    handlers['debug'].setLevel(logging.DEBUG)
    handlers['info'].setLevel(logging.INFO)
    handlers['error'].setLevel(logging.ERROR)
    handlers['critical'].setLevel(logging.CRITICAL)
    handlers['console'].setLevel(logging.INFO)
    # Apply formatter to all handlers
    formatter = CustomFormatter()
    for handler in handlers.values():
        handler.setFormatter(formatter)
    # Configure root logger
    logger = logging.getLogger("InsuranceLLM")
    logger.setLevel(logging.DEBUG)
    # Add all handlers
    for handler in handlers.values():
        logger.addHandler(handler)
    # Log startup information
    logger.info(f"Starting new session {timestamp}")
    logger.info(f"Log directory: {log_dir}")
    return logger

# Custom theme configuration
custom_theme = Theme({"info": "bold cyan","warning": "bold yellow", "error": "bold red","critical": "bold white on red","success": "bold green","timestamp": "bold magenta","metrics": "bold blue","memory": "bold yellow","performance": "bold cyan",})

console = Console(theme=custom_theme)

class PerformanceMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.tokens = 0
        self.response_times = []
        
    def update(self, tokens: int, response_time: float = None):
        self.tokens += tokens
        if response_time:
            self.response_times.append(response_time)
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.elapsed_time if self.elapsed_time > 0 else 0
    
    @property
    def average_response_time(self) -> float:
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0

class InsuranceLLM:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.llm_ctx: Optional[Llama] = None
        self.metrics = PerformanceMetrics()
        self.logger = setup_logging()
        
        nvidia_llama3_chatqa_system = (
            "This is a chat between a user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. "
            "The assistant should also indicate when the answer cannot be found in the context. "
        )
        enhanced_system_message = (
            "You are an expert and experienced from the Insurance domain with extensive insurance knowledge and "
            "professional writer skills, especially about insurance policies. "
            "Your name is OpenInsuranceLLM, and you were developed by Raj Maharajwala. "
            "You are willing to help answer the user's query with a detailed explanation. "
            "In your explanation, leverage your deep insurance expertise, such as relevant insurance policies, "
            "complex coverage plans, or other pertinent insurance concepts. Use precise insurance terminology while "
            "still aiming to make the explanation clear and accessible to a general audience."
        )
        self.full_system_message = nvidia_llama3_chatqa_system + enhanced_system_message

    # def check_metal_support(self):
    #     try:
    #         import Metal
    #         devices = Metal.MTLCopyAllDevices()
    #         if len(devices) > 0:
    #             console.print(f"[green]Metal devices available: {len(devices)}[/green]")
    #             return True
    #         console.print("[yellow]No Metal devices found[/yellow]")
    #         return False
    #     except ImportError:
    #         console.print("[red]Metal support not available[/red]")
    #         return False

    @contextmanager
    def timer(self, description: str):
        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        self.logger.info(f"{description}: {elapsed_time:.2f}s")

    def download_model(self) -> str:
        try:
            with console.status("[bold green]Downloading model..."):
                model_path = hf_hub_download(
                    self.config.model_name,
                    filename=self.config.model_file,
                    local_dir=os.path.join(os.getcwd(), 'gguf_dir')
                )
            self.logger.info(f"Model downloaded successfully to {model_path}")
            return model_path
        except Exception as e:
            self.logger.error(f"Error downloading model: {str(e)}")
            raise

    def load_model(self) -> None:
        try:
            # self.check_metal_support()
            quantized_path = os.path.join(os.getcwd(), "gguf_dir")
            directory = Path(quantized_path)
            
            try:
                model_path = str(list(directory.glob(self.config.model_file))[0])
            except IndexError:
                model_path = self.download_model()
            
            with console.status("[bold green]Loading model..."):
                self.llm_ctx = Llama(
                    model_path=model_path,
                    n_gpu_layers=self.config.n_gpu_layers,
                    n_ctx=self.config.n_ctx,
                    n_batch=self.config.n_batch,
                    num_beams=self.config.num_beams,
                    verbose=self.config.verbose,
                    use_mlock=self.config.use_mlock,
                    use_mmap=self.config.verbose,
                    offload_kqv=self.config.verbose
                )
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def get_prompt(self, question: str, context: str = "") -> str:
        if context:
            return (
                f"System: {self.full_system_message}\n\n"
                f"User: Context: {context}\nQuestion: {question}\n\n"
                "Assistant:"
            )
        return (
            f"System: {self.full_system_message}\n\n"
            f"User: Question: {question}\n\n"
            "Assistant:"
        )

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        if not self.llm_ctx:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            response = {"text": "", "tokens": 0}
            console.print("[bold cyan]Assistant:[/bold cyan]", end=" ")
            
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
                response["text"] += text_chunk
                response["tokens"] += 1
                console.print(text_chunk, end="", markup=False)
            
            console.print()
            return response
            
        except RuntimeError as e:
            if "llama_decode returned -3" in str(e):
                self.logger.error("Memory allocation failed. Try reducing context window or batch size")
            raise

    def print_metrics(self, response_tokens: int, memory_used: float):
        try:
            console.print("\n[dim]Performance Metrics:[/dim]")
            console.print(f"[dim]Memory usage: {memory_used:.2f} MB[/dim]")
            console.print(f"[dim]Tokens generated: {response_tokens}[/dim]")
            console.print(f"[dim]Average tokens/sec: {self.metrics.tokens_per_second:.2f}[/dim]")
            console.print(f"[dim]Total tokens: {self.metrics.tokens}[/dim]")
            console.print(f"[dim]Total time: {self.metrics.elapsed_time:.2f}s[/dim]\n")
        except Exception as e:
            self.logger.error(f"Error printing metrics: {str(e)}")
            print(f"\nPerformance Metrics:")
            print(f"Memory usage: {memory_used:.2f} MB")
            print(f"Tokens generated: {response_tokens}")
            print(f"Average tokens/sec: {self.metrics.tokens_per_second:.2f}")
            print(f"Total tokens: {self.metrics.tokens}")
            print(f"Total time: {self.metrics.elapsed_time:.2f}s\n")

    def run_inference_loop(self):
        try:
            self.load_model()
            console.print("\n[bold green]Welcome to Open-Insurance-LLM![/bold green]")
            console.print("Enter your questions (type 'exit' to quit)\n")
            console.print("Optional: You can provide context by typing 'context:' followed by your context, then 'question:' followed by your question\n")
            
            while True:
                try:
                    user_input = console.input("[bold cyan]User:[/bold cyan] ").strip()
                    
                    if user_input.lower() == "exit":
                        console.print("\n[bold green]Thank you for using OpenInsuranceLLM![/bold green]")
                        break
                    
                    context = ""
                    question = user_input
                    if "context:" in user_input.lower() and "question:" in user_input.lower():
                        parts = user_input.split("question:", 1)
                        context = parts[0].replace("context:", "").strip()
                        question = parts[1].strip()
                    
                    prompt = self.get_prompt(question, context)
                    
                    with self.timer("Response generation"):
                        response = self.generate_response(prompt)
                    
                    self.metrics.update(response["tokens"])
                    memory_used = psutil.Process().memory_info().rss / 1024 / 1024
                    self.print_metrics(response["tokens"], memory_used)
                    
                except KeyboardInterrupt:
                    console.print("\n[yellow]Input interrupted. Type 'exit' to quit.[/yellow]")
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing input: {str(e)}")
                    console.print(f"\n[red]Error processing input: {str(e)}[/red]")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Fatal error in inference loop: {str(e)}")
            console.print(f"\n[red]Fatal error: {str(e)}[/red]")
        finally:
            if self.llm_ctx:
                del self.llm_ctx

def main():
    try:
        config = ModelConfig()
        llm = InsuranceLLM(config)
        llm.run_inference_loop()
    except KeyboardInterrupt:
        console.print("\n[yellow]Program interrupted by user[/yellow]")
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        console.print(f"\n[red]Fatal error: {str(e)}[/red]")

if __name__ == "__main__":
    main()