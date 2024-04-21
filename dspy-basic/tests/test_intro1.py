# Test the intro1.py example. See https://dspy-docs.vercel.app/docs/tutorials/rag
import pytest
import dspy
from dspy_basic.intro1 import load_and_configure
from dspy_basic.intro1 import GenerateAnswer
from dspy_basic.intro1 import RAG
from dspy_basic.intro1 import compile_pipeline
from dspy_basic.intro1 import load_dataset
from dspy_basic.intro1 import validate_context_and_answer
import pydantic

@pytest.fixture
def trainset():
    return load_dataset()[0]

@pytest.fixture
def devset():
    return load_dataset()[1]

def test_load_and_configure_runs():
    assert load_and_configure()
    
def test_load_and_configure_returns_tuple():
    assert isinstance(load_and_configure(), tuple)
    assert len(load_and_configure()) == 2
    
def test_load_and_configure_returns_ollamalocal():
    assert isinstance(load_and_configure()[0], dspy.OllamaLocal)
    
def test_signature_type():
    assert isinstance(GenerateAnswer, dspy.signatures.signature.SignatureMeta)
    
def test_signature_fields_info():
    assert isinstance(GenerateAnswer.model_fields['context'], pydantic.fields.FieldInfo)
    assert isinstance(GenerateAnswer.model_fields['question'], pydantic.fields.FieldInfo)
    assert isinstance(GenerateAnswer.model_fields['answer'], pydantic.fields.FieldInfo)
    
def test_signature_definition():
    assert GenerateAnswer.signature == "context, question -> answer"
    
def test_pipeline_type():
    assert isinstance(RAG, dspy.primitives.program.ProgramMeta)
    
def test_pipeline_compilation():
    assert compile_pipeline(pipeline=RAG(), metric_callable=validate_context_and_answer, trainset=trainset)