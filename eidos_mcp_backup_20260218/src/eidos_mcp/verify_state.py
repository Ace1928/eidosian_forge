from .state import gis, audit, llm, agent, refactor, type_forge
from eidosian_core import eidosian

@eidosian()
def main():
    print(f"GIS: {gis}")
    print(f"Audit: {audit}")
    print(f"LLM: {llm}")
    print(f"Agent: {agent}")
    print(f"Refactor: {refactor}")
    print(f"Type: {type_forge}")

if __name__ == "__main__":
    main()
