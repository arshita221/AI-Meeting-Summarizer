"""
Data structures for the structured meeting report, plus safe parsing of the
LLM's JSON output into them.

Deliberately dependency-free (plain dataclasses instead of a validation
library) to keep this MVP's dependency list minimal, as required by the
project brief.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

NOT_SPECIFIED = "Not specified"

_VALID_PRIORITIES = {"high", "medium", "low", "not specified"}


class SummaryParsingError(Exception):
    """Raised when the LLM's JSON response cannot be parsed/validated."""


def _clean_str(value: object, default: str = "") -> str:
    """Coerce a value to a trimmed string, falling back to `default`."""
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _clean_str_list(value: object) -> list[str]:
    """Coerce a value into a list of non-empty, trimmed strings."""
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = _clean_str(item)
        if text:
            result.append(text)
    return result


@dataclass
class ActionItem:
    """A single task assigned or committed to during the meeting."""

    task: str
    owner: str = NOT_SPECIFIED
    deadline: str = NOT_SPECIFIED
    priority: str = NOT_SPECIFIED

    @staticmethod
    def from_dict(raw: dict) -> "ActionItem | None":
        task = _clean_str(raw.get("task"))
        if not task:
            return None  # An action item with no task text is not usable.

        owner = _clean_str(raw.get("owner"), NOT_SPECIFIED)
        deadline = _clean_str(raw.get("deadline"), NOT_SPECIFIED)
        priority_raw = _clean_str(raw.get("priority"), NOT_SPECIFIED)
        priority = priority_raw if priority_raw.lower() in _VALID_PRIORITIES else NOT_SPECIFIED
        # Normalize casing for consistent display (e.g. "high" -> "High").
        priority = priority if priority == NOT_SPECIFIED else priority.capitalize()

        return ActionItem(task=task, owner=owner, deadline=deadline, priority=priority)


@dataclass
class MeetingSummary:
    """The full structured output produced from a meeting transcript."""

    summary: str
    discussion_points: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    follow_up_items: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "discussion_points": self.discussion_points,
            "decisions": self.decisions,
            "action_items": [
                {
                    "task": item.task,
                    "owner": item.owner,
                    "deadline": item.deadline,
                    "priority": item.priority,
                }
                for item in self.action_items
            ],
            "follow_up_items": self.follow_up_items,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self, meeting_title: str = "Meeting Report") -> str:
        lines = [f"# {meeting_title}", "", "## Executive Summary", self.summary, ""]

        lines.append("## Key Discussion Points")
        lines.extend(f"- {p}" for p in self.discussion_points) if self.discussion_points else lines.append(
            "- None recorded."
        )
        lines.append("")

        lines.append("## Key Decisions")
        lines.extend(f"- {d}" for d in self.decisions) if self.decisions else lines.append("- None recorded.")
        lines.append("")

        lines.append("## Action Items")
        if self.action_items:
            lines.append("| Task | Owner | Deadline | Priority |")
            lines.append("|---|---|---|---|")
            for item in self.action_items:
                lines.append(f"| {item.task} | {item.owner} | {item.deadline} | {item.priority} |")
        else:
            lines.append("- None recorded.")
        lines.append("")

        lines.append("## Follow-up / Unresolved Items")
        lines.extend(f"- {f}" for f in self.follow_up_items) if self.follow_up_items else lines.append(
            "- None recorded."
        )
        lines.append("")

        return "\n".join(lines)


def parse_meeting_summary(raw_json: str) -> MeetingSummary:
    """
    Safely parse and validate an LLM JSON response into a MeetingSummary.

    Never trusts the LLM's output shape blindly: missing keys fall back to
    sensible defaults ("Not specified" / empty lists), malformed entries
    are dropped rather than crashing the app, and a summary with no usable
    text at all raises SummaryParsingError so the caller can retry safely.
    """
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SummaryParsingError("The AI response was not valid JSON.") from exc

    if not isinstance(data, dict):
        raise SummaryParsingError("The AI response was not a JSON object.")

    summary_text = _clean_str(data.get("summary"))
    if not summary_text:
        raise SummaryParsingError("The AI response did not include a summary.")

    action_items = []
    for raw_item in data.get("action_items") or []:
        if isinstance(raw_item, dict):
            item = ActionItem.from_dict(raw_item)
            if item is not None:
                action_items.append(item)

    return MeetingSummary(
        summary=summary_text,
        discussion_points=_clean_str_list(data.get("discussion_points")),
        decisions=_clean_str_list(data.get("decisions")),
        action_items=action_items,
        follow_up_items=_clean_str_list(data.get("follow_up_items")),
    )
