# Test the intro1.py example. See https://dspy-docs.vercel.app/docs/tutorials/rag
import pytest
import dspy
from dspy_basic.intro1 import GenerateAnswer
from dspy_basic.intro1 import RAG
from dspy_basic.intro1 import (
    compile_pipeline, 
    load_dataset, 
    get_config,
    load_and_configure,
    validate_context_and_answer)


import pydantic

@pytest.fixture
def trainset():
    return load_dataset()[0]

@pytest.fixture
def devset():
    return load_dataset()[1]

@pytest.fixture
def example_question():
    return "What is quantum field theory?"

@pytest.fixture
def compiled_pipeline(trainset):
    return compile_pipeline(
        pipeline=RAG(), 
        metric_callable=validate_context_and_answer, 
        trainset=trainset
        )

def test_get_config():
    assert get_config(config_path='configs/dev.json')
    
@pytest.fixutre
def config_dict():
    return get_config(config_path='configs/dev.json')
    

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
    
def test_fixtures():
    assert trainset
    assert devset
    
def test_pipeline_compilation(compiled_pipeline):
    assert compiled_pipeline
    
def test_compiled_pipeline_call(compiled_pipeline, example_question):
    assert compiled_pipeline(example_question)
    # assert isinstance(compiled_pipeline(example_question).answer, str) # TODO: figure out why this fails
    # assert isinstance(compiled_pipeline(example_question).context, str) # TODO: figure out why this fails