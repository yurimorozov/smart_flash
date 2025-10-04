# English B2 Phrases Generator

This Python script generates 50 English phrases or sentences for B2 level learning, focused on business communications, computers, and software development topics.

## Features

- **Multiple AI Providers**: Supports OpenAI ChatGPT, Anthropic Claude, and local Ollama models
- **Focused Topics**: Business communications, computers, and software development
- **B2 Level**: Intermediate-upper intermediate complexity
- **Formatted Output**: Saves phrases to a timestamped text file
- **Command Line Interface**: Easy to use with various options

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys (choose one or more providers):

### For OpenAI ChatGPT:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### For Anthropic Claude:
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
```

### For local Ollama:
- Install Ollama from https://ollama.ai/
- Pull a model: `ollama pull llama3`
- Start Ollama service: `ollama serve`

## Usage

### Basic usage (OpenAI ChatGPT):
```bash
python english_phrases_generator.py
```

### Using different providers:
```bash
# Use Anthropic Claude
python english_phrases_generator.py --provider anthropic

# Use local Ollama
python english_phrases_generator.py --provider ollama

# Use specific model
python english_phrases_generator.py --provider openai --model gpt-4
```

### Additional options:
```bash
# Specify output file
python english_phrases_generator.py --output my_phrases.txt

# Use custom API key
python english_phrases_generator.py --api-key your-key-here

# Use custom Ollama URL
python english_phrases_generator.py --provider ollama --base-url http://localhost:11434
```

## Examples

### Example output file structure:
```
English B2 Level Phrases for Learning
==================================================
Generated on: 2024-10-04 15:30:45
Topics: Business Communications, Computers, Software Development

 1. I need to schedule a meeting with the development team to discuss the project timeline.
 2. The software requires regular updates to maintain optimal performance.
 3. Could you please review the quarterly report before tomorrow's presentation?
 4. We're experiencing some technical difficulties with the database connection.
 5. The user interface needs to be more intuitive for better user experience.
...
```

## API Keys Setup

### OpenAI
1. Go to https://platform.openai.com/
2. Create an account and get an API key
3. Set the environment variable: `export OPENAI_API_KEY="sk-..."`

### Anthropic
1. Go to https://console.anthropic.com/
2. Create an account and get an API key
3. Set the environment variable: `export ANTHROPIC_API_KEY="sk-ant-..."`

### Ollama (Local)
1. Install from https://ollama.ai/
2. Run: `ollama pull llama3` (or another model)
3. Start service: `ollama serve`
4. No API key needed for local usage

## Troubleshooting

### Common Issues:

1. **"API key is required" error**:
   - Make sure you've set the correct environment variable
   - Or use `--api-key` parameter

2. **"Connection error" with Ollama**:
   - Check if Ollama service is running: `ollama serve`
   - Verify the model is installed: `ollama list`

3. **Fewer than 50 phrases generated**:
   - This can happen with some models; the script will warn you
   - Try running again or using a different provider/model

4. **Rate limiting errors**:
   - Wait a moment and try again
   - Consider using a different provider

## Customization

You can modify the script to:
- Change the topic focus by editing `prompt_template`
- Adjust the language level (A2, B1, C1, etc.)
- Add new AI providers
- Change output formatting

## License


This script is provided as-is for educational purposes.