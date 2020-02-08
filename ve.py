from pgmpy.inference import VariableElimination
import re

class VarElimination(VariableElimination):
    calls = [0]
    def __init__(self, bayes_net):
        super().__init__(bayes_net)
        self.calls[0] += 1
    
    @classmethod
    def get_calls(cls):
        return cls.calls[0]