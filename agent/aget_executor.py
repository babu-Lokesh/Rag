from agent.chains import get_agent
import sys,os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

agent = get_agent()
question = " what is by billing date"
agent.invoke(question)