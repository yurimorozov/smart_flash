#!/usr/bin/env python3
"""
English Phrases Generator for B2 Level Learning

This script uses LangGraph and structured AI models to generate
50 phrases or sentences for learning English at B2 level, focusing on:
- Business communications
- Computers
- Software development

Usage:
    python english_phrases_generator.py
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional, TypedDict, Annotated, Literal, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig


class TopicCategory(str, Enum):
    """Enumeration of phrase topic categories"""
    BUSINESS = "business"
    COMPUTERS = "computers"
    SOFTWARE = "software"


class DifficultyLevel(str, Enum):
    """Enumeration of language difficulty levels"""
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"


class Phrase(BaseModel):
    """Individual phrase model with metadata"""
    text: str = Field(..., description="The actual phrase or sentence")
    topic: TopicCategory = Field(..., description="Topic category of the phrase")
    complexity_score: int = Field(ge=1, le=10, description="Complexity score from 1-10")
    key_vocabulary: List[str] = Field(default_factory=list, description="Key vocabulary words in the phrase")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Phrase must be at least 10 characters long")
        if len(v.strip()) > 200:
            raise ValueError("Phrase must be no longer than 200 characters")
        return v.strip()


class PhrasesCollection(BaseModel):
    """Collection of generated phrases with metadata"""
    phrases: List[Phrase] = Field(..., description="List of generated phrases")
    level: DifficultyLevel = Field(default=DifficultyLevel.B2, description="Target language level")
    total_count: int = Field(..., description="Total number of phrases")
    generation_timestamp: datetime = Field(default_factory=datetime.now)
    
    @field_validator('phrases')
    @classmethod
    def validate_phrase_count(cls, v: List[Phrase]) -> List[Phrase]:
        if len(v) < 40:
            raise ValueError(f"Expected at least 40 phrases, got {len(v)}")
        return v
    
    @field_validator('total_count')
    @classmethod
    def validate_total_count(cls, v: int, info) -> int:
        # Note: In Pydantic V2, we use info.data instead of values
        if 'phrases' in info.data and v != len(info.data['phrases']):
            raise ValueError("total_count must match the actual number of phrases")
        return v


class GraphState(TypedDict):
    """State for the LangGraph workflow"""
    provider: str
    model: str
    api_key: Optional[str]
    target_count: int
    level: DifficultyLevel
    phrases_collection: Optional[PhrasesCollection]
    raw_response: Optional[str]
    retry_count: int
    errors: List[str]
    validation_passed: bool


class LangChainProvider:
    """Unified provider using LangChain for different AI models"""
    
    def __init__(self, provider_type: str, api_key: str, model: str):
        self.provider_type = provider_type.lower()
        self.api_key = api_key
        self.model = model
        self.llm = self._create_llm()
        
    def _create_llm(self):
        """Create appropriate LangChain LLM instance"""
        if self.provider_type == "openai":
            return ChatOpenAI(
                api_key=self.api_key,
                model=self.model,
                temperature=0.7,
                max_tokens=2000
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider_type}")
    
    def generate_structured_phrases(self, system_prompt: str, user_prompt: str) -> PhrasesCollection:
        """Generate phrases with structured output"""
        parser = PydanticOutputParser(pydantic_object=PhrasesCollection)
        
        # Add format instructions to the prompt
        format_instructions = parser.get_format_instructions()
        enhanced_user_prompt = f"{user_prompt}\n\n{format_instructions}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=enhanced_user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return parser.parse(response.content)


# LangGraph workflow nodes
def initialize_state(state: GraphState) -> GraphState:
    """Initialize the graph state with default values"""
    state["retry_count"] = 0
    state["errors"] = []
    state["validation_passed"] = False
    state["phrases_collection"] = None
    state["raw_response"] = None
    return state

def get_api_key(api_path):
    token_file_path = os.environ.get(api_path)

    with open(token_file_path, 'r') as f:
        api_token = f.read().strip()    
    return api_token

def generate_phrases_node(state: GraphState) -> GraphState:
    """Generate phrases using the specified AI provider"""
    try:
        # Get API key
        api_key = get_api_key(api_path="OPENAI_TOKEN_PATH")
        
        if not api_key:
            raise ValueError(f"{state['provider'].upper()}_TOKEN_PATH is required")
        
        # Create provider
        provider = LangChainProvider(state["provider"], api_key, state["model"])
        
        # Define prompts
        system_prompt = """
You are an expert English language teacher. 
Your task is to generate practical, real-world phrases for business and technology contexts.
        """
        
        user_prompt = f"""Generate exactly {state['target_count']} English phrases or sentences for {state['level'].value} level learners.

Focus on these topics:
1. Business communications: emails, meetings, presentations, client interactions
2. Computers and technology: troubleshooting, system administration, user interfaces
3. Software development: coding practices, debugging, version control, testing

Requirements:
- B2 level complexity (intermediate-upper intermediate)
- Practical and useful for real-world situations
- Mix of formal and semi-formal register
- Vary sentence structures and lengths
- Include relevant technical and business vocabulary

For each phrase, also identify:
- The main topic category (business, computers, or software)
- Complexity score (1-10, targeting 5-7)
- 1-3 key vocabulary words from the phrase"""
        
        # Generate structured output
        phrases_collection = provider.generate_structured_phrases(system_prompt, user_prompt)
        state["phrases_collection"] = phrases_collection
        state["raw_response"] = "Generated using structured output"
        
        print(f"✓ Generated {len(phrases_collection.phrases)} phrases using {state['provider']}")
        
    except Exception as e:
        error_msg = f"Generation failed: {str(e)}"
        state["errors"].append(error_msg)
        print(f"✗ {error_msg}")
    
    return state


def validate_phrases_node(state: GraphState) -> GraphState:
    """Validate the generated phrases"""
    if not state["phrases_collection"]:
        state["validation_passed"] = False
        return state
    
    collection = state["phrases_collection"]
    errors = []
    
    # Check phrase count
    if len(collection.phrases) < 40:
        errors.append(f"Insufficient phrases: {len(collection.phrases)} (expected {state['target_count']})")
    
    # Check phrase quality
    valid_phrases = 0
    for i, phrase in enumerate(collection.phrases, 1):
        try:
            # Validate phrase length
            if len(phrase.text.strip()) < 10:
                errors.append(f"Phrase {i} too short: '{phrase.text[:30]}...'")
            elif len(phrase.text.strip()) > 200:
                errors.append(f"Phrase {i} too long: '{phrase.text[:30]}...'")
            else:
                valid_phrases += 1
                
        except Exception as e:
            errors.append(f"Phrase {i} validation error: {e}")
    
    # Check if we have enough valid phrases
    if valid_phrases < 40:
        errors.append(f"Too few valid phrases: {valid_phrases}/40")
    
    if errors:
        state["errors"].extend(errors[:5])  # Limit error list
        state["validation_passed"] = False
        print(f"✗ Validation failed: {len(errors)} issues found")
    else:
        state["validation_passed"] = True
        print(f"✓ Validation passed: {valid_phrases} valid phrases")
    
    return state


def retry_decision_node(state: GraphState) -> str:
    """Decide whether to retry generation or proceed"""
    if state["validation_passed"]:
        return "save_output"
    
    if state["retry_count"] < 2:
        state["retry_count"] += 1
        print(f"🔄 Retrying generation (attempt {state['retry_count'] + 1}/3)")
        return "generate_phrases"
    else:
        print("❌ Max retries reached")
        return "save_output"


def save_output_node(state: GraphState) -> GraphState:
    """Save the generated phrases to file"""
    if not state["phrases_collection"]:
        state["errors"].append("No phrases to save")
        return state
    
    try:
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"english_b2_phrases_{timestamp}.txt"
        
        collection = state["phrases_collection"]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("English B2 Level Phrases for Learning\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {collection.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Level: {collection.level.value}\n")
            f.write(f"Provider: {state['provider']} ({state['model']})\n")
            f.write("Topics: Business Communications, Computers, Software Development\n\n")
            
            # Group phrases by topic
            business_phrases = [p for p in collection.phrases if p.topic == TopicCategory.BUSINESS]
            computer_phrases = [p for p in collection.phrases if p.topic == TopicCategory.COMPUTERS]
            software_phrases = [p for p in collection.phrases if p.topic == TopicCategory.SOFTWARE]
            
            def write_section(title: str, phrases: List[Phrase], start_num: int):
                if phrases:
                    f.write(f"{title}\n")
                    f.write("-" * len(title) + "\n")
                    for i, phrase in enumerate(phrases, start_num):
                        f.write(f"{i:2d}. {phrase.text}\n")
                        if phrase.key_vocabulary:
                            f.write(f"    Key words: {', '.join(phrase.key_vocabulary)}\n")
                        f.write(f"    Complexity: {phrase.complexity_score}/10\n\n")
                    return start_num + len(phrases)
                return start_num
            
            num = 1
            num = write_section("BUSINESS COMMUNICATIONS", business_phrases, num)
            num = write_section("COMPUTERS & TECHNOLOGY", computer_phrases, num)
            num = write_section("SOFTWARE DEVELOPMENT", software_phrases, num)
            
            f.write(f"Total phrases: {len(collection.phrases)}\n")
            if state["errors"]:
                f.write(f"\nGeneration issues: {len(state['errors'])}\n")
                for error in state["errors"]:
                    f.write(f"- {error}\n")
        
        state["output_file"] = output_file
        print(f"✓ Saved to: {output_file}")
        
    except Exception as e:
        error_msg = f"Failed to save output: {e}"
        state["errors"].append(error_msg)
        print(f"✗ {error_msg}")
    
    return state


class EnglishPhrasesGenerator:
    """Main class for generating English phrases using LangGraph"""
    
    def __init__(self):
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("initialize", initialize_state)
        workflow.add_node("generate_phrases", generate_phrases_node)
        workflow.add_node("validate", validate_phrases_node)
        workflow.add_node("save_output", save_output_node)
        
        # Add edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "generate_phrases")
        workflow.add_edge("generate_phrases", "validate")
        
        # Add conditional edge for retry logic
        workflow.add_conditional_edges(
            "validate",
            retry_decision_node,
            {
                "generate_phrases": "generate_phrases",
                "save_output": "save_output"
            }
        )
        
        workflow.add_edge("save_output", END)
        
        return workflow.compile()
    
    def generate_phrases(self, provider: str = "openai", model: str = None, 
                        api_key: str = None, target_count: int = 50) -> Dict[str, Any]:
        """Generate phrases using the LangGraph workflow"""
        
        # Set default models
        if not model:
            model = "gpt-3.5-turbo" if provider == "openai" else "claude-3-sonnet-20240229"
        
        # Initial state
        initial_state: GraphState = {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "target_count": target_count,
            "level": DifficultyLevel.B2,
            "phrases_collection": None,
            "raw_response": None,
            "retry_count": 0,
            "errors": [],
            "validation_passed": False
        }
        
        print(f"🚀 Starting phrase generation with {provider} ({model})")
        
        # Run the workflow
        final_state = self.graph.invoke(initial_state)
        
        # Prepare result
        result = {
            "success": final_state["validation_passed"],
            "phrases_count": len(final_state["phrases_collection"].phrases) if final_state["phrases_collection"] else 0,
            "output_file": final_state.get("output_file"),
            "errors": final_state["errors"],
            "retry_count": final_state["retry_count"],
            "phrases_collection": final_state["phrases_collection"]
        }
        
        return result


def main():
    parser = argparse.ArgumentParser(description="Generate English B2 level phrases for learning")
    parser.add_argument("--provider", choices=["openai", "anthropic"], 
                       default="openai", help="AI provider to use")
    parser.add_argument("--model", help="Specific model to use")
    parser.add_argument("--api-key", help="API key (overrides environment variable)")
    parser.add_argument("--count", type=int, default=50, help="Number of phrases to generate")
    
    args = parser.parse_args()
    
    print("🎯 English B2 Phrases Generator with LangGraph")
    print("=" * 50)
    
    generator = EnglishPhrasesGenerator()
    
    try:
        # Generate phrases using LangGraph workflow
        result = generator.generate_phrases(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            target_count=args.count
        )
        
        # Print results
        print(f"\n📊 Generation Results:")
        print(f"Success: {'✓' if result['success'] else '✗'}")
        print(f"Phrases generated: {result['phrases_count']}")
        print(f"Retry attempts: {result['retry_count']}")
        
        if result['output_file']:
            print(f"Output file: {result['output_file']}")
        
        if result['errors']:
            print(f"\n⚠️  Issues encountered:")
            for error in result['errors']:
                print(f"  - {error}")
        
        # Display preview if successful
        if result['success'] and result['phrases_collection']:
            print(f"\n🔍 Preview (first 5 phrases):")
            for i, phrase in enumerate(result['phrases_collection'].phrases[:5], 1):
                print(f"{i:2d}. {phrase.text}")
                print(f"    Topic: {phrase.topic.value} | Complexity: {phrase.complexity_score}/10")
                if phrase.key_vocabulary:
                    print(f"    Key words: {', '.join(phrase.key_vocabulary)}")
                print()
            
            if len(result['phrases_collection'].phrases) > 5:
                print("    ... (see output file for complete list)")
        
        return 0 if result['success'] else 1
            
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())