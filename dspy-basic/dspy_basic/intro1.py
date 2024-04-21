# See https://dspy-docs.vercel.app/docs/tutorials/rag
import dspy
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShot


def load_and_configure():
    # load models
    llama2 = dspy.OllamaLocal(model="llama2")
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    # configure
    dspy.settings.configure(lm=llama2, rm=colbertv2_wiki17_abstracts)
    return llama2, colbertv2_wiki17_abstracts

# Load the train and development set.
def load_dataset():
    dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)
    # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
    trainset = [x.with_inputs('question') for x in dataset.train]
    devset = [x.with_inputs('question') for x in dataset.dev]
    #print("len(trainset):", len(trainset))
    #print("len(devset):", len(devset))
    return trainset, devset

# Define the signature for QA in the RAG system.
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
    
# Define the pipeline
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

def compile_pipeline(pipeline=RAG(), metric_callable=validate_context_and_answer, trainset=None):
    # Set up a basic teleprompter, which will compile our RAG program.
    teleprompter = BootstrapFewShot(metric=metric_callable)
    # Compile!
    compiled_rag = teleprompter.compile(pipeline, trainset=trainset)
    return compiled_rag
    

if __name__ == '__main__':
    # load models
    llama2, colbertv2_wiki17_abstracts = load_and_configure()
    trainset, devset = load_dataset()
    print("trainset[0]:", trainset[0])
    print("devset[0]:", devset[0])
    
    compile_pipeline(pipeline=RAG(), metric_callable=validate_context_and_answer, trainset=trainset)