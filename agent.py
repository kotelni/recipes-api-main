"""
Please provide the full URL to your recipes-api GitHub repository below.
"""

import asyncio
import os
import dotenv
from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentWorkflow,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from typing import Any
from github import Auth, Github

#REPO = "https://github.com/kotelni/recipes-api-main.git"

#_repo_name = REPO.split("/")[-1].replace(".git", "")
#_username = REPO.split("/")[-2]
#FULL_REPO_NAME = f"{_username}/{_repo_name}"
FULL_REPO_NAME = os.getenv("REPOSITORY")

dotenv.load_dotenv()

_token = os.getenv("GITHUB_TOKEN")
git = Github(auth=Auth.Token(_token)) if _token else Github()
repo = git.get_repo(FULL_REPO_NAME)


def get_pr_details(pr_number: int) -> dict[str, Any]:
    """Fetch details of a pull request given its number.

    Returns a dictionary with the PR author (user), title, body, diff_url,
    state, head_sha (the SHA of the latest commit) and commit_shas (the list
    of all commit SHAs in the pull request). Use the commit SHAs with the
    get_commit_details tool to find out which files were changed.
    """
    pr = repo.get_pull(pr_number)
    commit_shas = [c.sha for c in pr.get_commits()]
    return {
        "number": pr.number,
        "user": pr.user.login,
        "title": pr.title,
        "body": pr.body,
        "diff_url": pr.diff_url,
        "state": pr.state,
        "head_sha": pr.head.sha,
        "commit_shas": commit_shas,
    }


def get_file_content(file_path: str) -> str:
    """Fetch the contents of a file from the repository given its path.

    The path is relative to the repository root, e.g. "app/models.py" or
    "pytest.ini". Returns the decoded text contents of the file.
    """
    content = repo.get_contents(file_path)
    if isinstance(content, list):
        raise ValueError(f"'{file_path}' is a directory, not a file.")
    return content.decoded_content.decode("utf-8")


def get_commit_details(commit_sha: str) -> list[dict[str, Any]]:
    """Fetch details of a commit given its SHA.

    Returns a list of the files changed in the commit. Each entry contains
    the filename, status, number of additions, deletions and changes, and
    the patch (diff) for that file.
    """
    commit = repo.get_commit(commit_sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append(
            {
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch,
            }
        )
    return changed_files


def post_review_comment(pr_number: int, comment: str) -> str:
    """Post the final review comment on a pull request given its number.

    Submits the comment as a pull request review on GitHub. Returns a short
    confirmation message with the URL of the created review.
    """
    pr = repo.get_pull(pr_number)
    review = pr.create_review(body=comment)
    return f"Review posted on PR #{pr_number}: {review.html_url}"


#repo_url = REPO

# 1. Initialize the OpenAI provider
llm = OpenAI(
    model="gpt-5.6-sol",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)


# 2. State management tools
async def add_context_to_state(ctx: Context, context: str) -> str:
    """Useful for adding the gathered context (PR details, changed files, file
    contents) to the shared state so that other agents can use it."""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["gathered_contexts"] = context
    return "State updated with the gathered contexts."


async def add_comment_to_state(ctx: Context, review_comment: str) -> str:
    """Useful for saving the draft PR review comment to the shared state."""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["review_comment"] = review_comment
    return "State updated with the draft review comment."


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Useful for saving the final, approved PR review to the shared state."""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["final_review"] = final_review
    return "State updated with the final review."


# 3. Convert the GitHub functions into tools
pr_details_tool = FunctionTool.from_defaults(get_pr_details)
file_content_tool = FunctionTool.from_defaults(get_file_content)
commit_details_tool = FunctionTool.from_defaults(get_commit_details)
post_review_tool = FunctionTool.from_defaults(post_review_comment)

CONTEXT_AGENT_SYSTEM_PROMPT = """You are the context gathering agent. When gathering context, you MUST gather \n:
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any specifically requested by the user files; \n
  - Do not request any other than the changed files or the files specifically asked in the user's question; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent."""

COMMENTOR_AGENT_SYSTEM_PROMPT = """You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n
Ensure to do the following for a thorough review:
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
 - If you need any additional details, you must hand off to the Commentor Agent. \n
 - You should directly address the author. So your comments should sound like: \n
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review.
"""

REVIEW_AND_POSTING_AGENT_SYSTEM_PROMPT = """You are the Review and Posting agent. You must use the CommentorAgent to create a review comment.
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub."""

# 4. Context gathering agent
context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description=(
        "Gathers all the needed context from the GitHub repository: pull request "
        "details, changed files in commits, and the contents of any user requested files. "
        "Saves the gathered context to the state and hands off to the CommentorAgent."
    ),
    system_prompt=CONTEXT_AGENT_SYSTEM_PROMPT,
    tools=[pr_details_tool, file_content_tool, commit_details_tool, add_context_to_state],
    can_handoff_to=["CommentorAgent"],
)

# 5. Commentor agent
commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment comment.",
    system_prompt=COMMENTOR_AGENT_SYSTEM_PROMPT,
    tools=[add_comment_to_state],
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

# 6. Review and posting agent
review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description=(
        "Checks the draft review comment against the review criteria, requests a "
        "rewrite from the CommentorAgent if needed, saves the final review to the "
        "state and posts it to the pull request on GitHub."
    ),
    system_prompt=REVIEW_AND_POSTING_AGENT_SYSTEM_PROMPT,
    tools=[add_final_review_to_state, post_review_tool],
    can_handoff_to=["CommentorAgent"],
)

# 7. Workflow orchestrating the agents
workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "review_comment": "",
        "final_review": "",
    },
)

# Shared state between workflow runs
context = Context(workflow_agent)

pr_number = os.getenv("PR_NUMBER")

async def main():
    #print("Enter the query: ")
    query = "Write a review for PR: " + pr_number
    prompt = RichPromptTemplate(query)
    handler = workflow_agent.run(prompt.format(), ctx=context)
    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
