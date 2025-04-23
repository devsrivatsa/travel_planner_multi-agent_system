from setuptools import setup, find_packages

setup(
    name="travel_planner",
    version="0.1",
    packages = find_packages(),
    install_requires = [
        "see-starlette",
        "uvicorn",
        "pydantic",
        "google-adk",
        "python-dotenv",
        "starlette",
        "google-genai"
    ]
)