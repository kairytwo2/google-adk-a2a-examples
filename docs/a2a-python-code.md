This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: **/*.py, **/*.md
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
.github/
  actions/
    spelling/
      advice.md
  PULL_REQUEST_TEMPLATE.md
src/
  a2a/
    auth/
      user.py
    client/
      __init__.py
      client.py
      errors.py
      helpers.py
    server/
      agent_execution/
        __init__.py
        agent_executor.py
        context.py
        request_context_builder.py
        simple_request_context_builder.py
      apps/
        __init__.py
        starlette_app.py
      events/
        __init__.py
        event_consumer.py
        event_queue.py
        in_memory_queue_manager.py
        queue_manager.py
      request_handlers/
        __init__.py
        default_request_handler.py
        jsonrpc_handler.py
        request_handler.py
        response_helpers.py
      tasks/
        __init__.py
        inmemory_push_notifier.py
        inmemory_task_store.py
        push_notifier.py
        result_aggregator.py
        task_manager.py
        task_store.py
        task_updater.py
      __init__.py
      context.py
    utils/
      __init__.py
      artifact.py
      errors.py
      helpers.py
      message.py
      task.py
      telemetry.py
    __init__.py
    types.py
tests/
  client/
    test_client.py
    test_errors.py
  server/
    agent_execution/
      test_context.py
    events/
      test_event_consumer.py
      test_event_queue.py
      test_inmemory_queue_manager.py
    request_handlers/
      test_jsonrpc_handler.py
    tasks/
      test_inmemory_task_store.py
      test_task_manager.py
      test_task_updater.py
    test_integration.py
  utils/
    test_helpers.py
    test_message.py
    test_telemetry.py
  README.md
  test_types.py
CHANGELOG.md
CODE_OF_CONDUCT.md
CONTRIBUTING.md
development.md
noxfile.py
README.md
SECURITY.md
```

# Files

## File: .github/actions/spelling/advice.md
````markdown
<!-- See https://github.com/check-spelling/check-spelling/wiki/Configuration-Examples%3A-advice --> <!-- markdownlint-disable MD033 MD041 -->
<details><summary>If the flagged items are :exploding_head: false positives</summary>

If items relate to a ...

- binary file (or some other file you wouldn't want to check at all).

  Please add a file path to the `excludes.txt` file matching the containing file.

  File paths are Perl 5 Regular Expressions - you can [test](https://www.regexplanet.com/advanced/perl/) yours before committing to verify it will match your files.

  `^` refers to the file's path from the root of the repository, so `^README\.md$` would exclude `README.md` (on whichever branch you're using).

- well-formed pattern.

  If you can write a [pattern](https://github.com/check-spelling/check-spelling/wiki/Configuration-Examples:-patterns) that would match it,
  try adding it to the `patterns.txt` file.

  Patterns are Perl 5 Regular Expressions - you can [test](https://www.regexplanet.com/advanced/perl/) yours before committing to verify it will match your lines.

  Note that patterns can't match multiline strings.

</details>

<!-- adoption information-->

:steam_locomotive: If you're seeing this message and your PR is from a branch that doesn't have check-spelling,
please merge to your PR's base branch to get the version configured for your repository.
````

## File: src/a2a/auth/user.py
````python
"""Authenticated user information."""

from abc import ABC, abstractmethod


class User(ABC):
    """A representation of an authenticated user."""

    @property
    @abstractmethod
    def is_authenticated(self) -> bool:
        """Returns whether the current user is authenticated."""

    @property
    @abstractmethod
    def user_name(self) -> str:
        """Returns the user name of the current user."""


class UnauthenticatedUser(User):
    """A representation that no user has been authenticated in the request."""

    @property
    def is_authenticated(self):
        return False

    @property
    def user_name(self) -> str:
        return ''
````

## File: src/a2a/client/errors.py
````python
"""Custom exceptions for the A2A client."""


class A2AClientError(Exception):
    """Base exception for A2A Client errors."""


class A2AClientHTTPError(A2AClientError):
    """Client exception for HTTP errors received from the server."""

    def __init__(self, status_code: int, message: str):
        """Initializes the A2AClientHTTPError.

        Args:
            status_code: The HTTP status code of the response.
            message: A descriptive error message.
        """
        self.status_code = status_code
        self.message = message
        super().__init__(f'HTTP Error {status_code}: {message}')


class A2AClientJSONError(A2AClientError):
    """Client exception for JSON errors during response parsing or validation."""

    def __init__(self, message: str):
        """Initializes the A2AClientJSONError.

        Args:
            message: A descriptive error message.
        """
        self.message = message
        super().__init__(f'JSON Error: {message}')
````

## File: src/a2a/server/tasks/task_store.py
````python
from abc import ABC, abstractmethod

from a2a.types import Task


class TaskStore(ABC):
    """Agent Task Store interface.

    Defines the methods for persisting and retrieving `Task` objects.
    """

    @abstractmethod
    async def save(self, task: Task):
        """Saves or updates a task in the store."""

    @abstractmethod
    async def get(self, task_id: str) -> Task | None:
        """Retrieves a task from the store by ID."""

    @abstractmethod
    async def delete(self, task_id: str):
        """Deletes a task from the store by ID."""
````

## File: src/a2a/server/__init__.py
````python
"""Server-side components for implementing an A2A agent."""
````

## File: src/a2a/__init__.py
````python
"""The A2A Python SDK."""
````

## File: tests/client/test_errors.py
````python
import pytest

from a2a.client import A2AClientError, A2AClientHTTPError, A2AClientJSONError


class TestA2AClientError:
    """Test cases for the base A2AClientError class."""

    def test_instantiation(self):
        """Test that A2AClientError can be instantiated."""
        error = A2AClientError('Test error message')
        assert isinstance(error, Exception)
        assert str(error) == 'Test error message'

    def test_inheritance(self):
        """Test that A2AClientError inherits from Exception."""
        error = A2AClientError()
        assert isinstance(error, Exception)


class TestA2AClientHTTPError:
    """Test cases for A2AClientHTTPError class."""

    def test_instantiation(self):
        """Test that A2AClientHTTPError can be instantiated with status_code and message."""
        error = A2AClientHTTPError(404, 'Not Found')
        assert isinstance(error, A2AClientError)
        assert error.status_code == 404
        assert error.message == 'Not Found'

    def test_message_formatting(self):
        """Test that the error message is formatted correctly."""
        error = A2AClientHTTPError(500, 'Internal Server Error')
        assert str(error) == 'HTTP Error 500: Internal Server Error'

    def test_inheritance(self):
        """Test that A2AClientHTTPError inherits from A2AClientError."""
        error = A2AClientHTTPError(400, 'Bad Request')
        assert isinstance(error, A2AClientError)

    def test_with_empty_message(self):
        """Test behavior with an empty message."""
        error = A2AClientHTTPError(403, '')
        assert error.status_code == 403
        assert error.message == ''
        assert str(error) == 'HTTP Error 403: '

    def test_with_various_status_codes(self):
        """Test with different HTTP status codes."""
        test_cases = [
            (200, 'OK'),
            (201, 'Created'),
            (400, 'Bad Request'),
            (401, 'Unauthorized'),
            (403, 'Forbidden'),
            (404, 'Not Found'),
            (500, 'Internal Server Error'),
            (503, 'Service Unavailable'),
        ]

        for status_code, message in test_cases:
            error = A2AClientHTTPError(status_code, message)
            assert error.status_code == status_code
            assert error.message == message
            assert str(error) == f'HTTP Error {status_code}: {message}'


class TestA2AClientJSONError:
    """Test cases for A2AClientJSONError class."""

    def test_instantiation(self):
        """Test that A2AClientJSONError can be instantiated with a message."""
        error = A2AClientJSONError('Invalid JSON format')
        assert isinstance(error, A2AClientError)
        assert error.message == 'Invalid JSON format'

    def test_message_formatting(self):
        """Test that the error message is formatted correctly."""
        error = A2AClientJSONError('Missing required field')
        assert str(error) == 'JSON Error: Missing required field'

    def test_inheritance(self):
        """Test that A2AClientJSONError inherits from A2AClientError."""
        error = A2AClientJSONError('Parsing error')
        assert isinstance(error, A2AClientError)

    def test_with_empty_message(self):
        """Test behavior with an empty message."""
        error = A2AClientJSONError('')
        assert error.message == ''
        assert str(error) == 'JSON Error: '

    def test_with_various_messages(self):
        """Test with different error messages."""
        test_messages = [
            'Malformed JSON',
            'Missing required fields',
            'Invalid data type',
            'Unexpected JSON structure',
            'Empty JSON object',
        ]

        for message in test_messages:
            error = A2AClientJSONError(message)
            assert error.message == message
            assert str(error) == f'JSON Error: {message}'


class TestExceptionHierarchy:
    """Test the exception hierarchy and relationships."""

    def test_exception_hierarchy(self):
        """Test that the exception hierarchy is correct."""
        assert issubclass(A2AClientError, Exception)
        assert issubclass(A2AClientHTTPError, A2AClientError)
        assert issubclass(A2AClientJSONError, A2AClientError)

    def test_catch_specific_exception(self):
        """Test that specific exceptions can be caught."""
        try:
            raise A2AClientHTTPError(404, 'Not Found')
        except A2AClientHTTPError as e:
            assert e.status_code == 404
            assert e.message == 'Not Found'

    def test_catch_base_exception(self):
        """Test that derived exceptions can be caught as base exception."""
        exceptions = [
            A2AClientHTTPError(404, 'Not Found'),
            A2AClientJSONError('Invalid JSON'),
        ]

        for raised_error in exceptions:
            try:
                raise raised_error
            except A2AClientError as e:
                assert isinstance(e, A2AClientError)


class TestExceptionRaising:
    """Test cases for raising and handling the exceptions."""

    def test_raising_http_error(self):
        """Test raising an HTTP error and checking its properties."""
        with pytest.raises(A2AClientHTTPError) as excinfo:
            raise A2AClientHTTPError(429, 'Too Many Requests')

        error = excinfo.value
        assert error.status_code == 429
        assert error.message == 'Too Many Requests'
        assert str(error) == 'HTTP Error 429: Too Many Requests'

    def test_raising_json_error(self):
        """Test raising a JSON error and checking its properties."""
        with pytest.raises(A2AClientJSONError) as excinfo:
            raise A2AClientJSONError('Invalid format')

        error = excinfo.value
        assert error.message == 'Invalid format'
        assert str(error) == 'JSON Error: Invalid format'

    def test_raising_base_error(self):
        """Test raising the base error."""
        with pytest.raises(A2AClientError) as excinfo:
            raise A2AClientError('Generic client error')

        assert str(excinfo.value) == 'Generic client error'


# Additional parametrized tests for more comprehensive coverage


@pytest.mark.parametrize(
    'status_code,message,expected',
    [
        (400, 'Bad Request', 'HTTP Error 400: Bad Request'),
        (404, 'Not Found', 'HTTP Error 404: Not Found'),
        (500, 'Server Error', 'HTTP Error 500: Server Error'),
    ],
)
def test_http_error_parametrized(status_code, message, expected):
    """Parametrized test for HTTP errors with different status codes."""
    error = A2AClientHTTPError(status_code, message)
    assert error.status_code == status_code
    assert error.message == message
    assert str(error) == expected


@pytest.mark.parametrize(
    'message,expected',
    [
        ('Missing field', 'JSON Error: Missing field'),
        ('Invalid type', 'JSON Error: Invalid type'),
        ('Parsing failed', 'JSON Error: Parsing failed'),
    ],
)
def test_json_error_parametrized(message, expected):
    """Parametrized test for JSON errors with different messages."""
    error = A2AClientJSONError(message)
    assert error.message == message
    assert str(error) == expected
````

## File: tests/server/agent_execution/test_context.py
````python
import uuid

from unittest.mock import Mock, patch

import pytest

from a2a.server.agent_execution import RequestContext
from a2a.types import (
    Message,
    MessageSendParams,
    Task,
)


class TestRequestContext:
    """Tests for the RequestContext class."""

    @pytest.fixture
    def mock_message(self):
        """Fixture for a mock Message."""
        return Mock(spec=Message, taskId=None, contextId=None)

    @pytest.fixture
    def mock_params(self, mock_message):
        """Fixture for a mock MessageSendParams."""
        return Mock(spec=MessageSendParams, message=mock_message)

    @pytest.fixture
    def mock_task(self):
        """Fixture for a mock Task."""
        return Mock(spec=Task, id='task-123', contextId='context-456')

    def test_init_without_params(self):
        """Test initialization without parameters."""
        context = RequestContext()
        assert context.message is None
        assert context.task_id is None
        assert context.context_id is None
        assert context.current_task is None
        assert context.related_tasks == []

    def test_init_with_params_no_ids(self, mock_params):
        """Test initialization with params but no task or context IDs."""
        with patch(
            'uuid.uuid4',
            side_effect=[
                uuid.UUID('00000000-0000-0000-0000-000000000001'),
                uuid.UUID('00000000-0000-0000-0000-000000000002'),
            ],
        ):
            context = RequestContext(request=mock_params)

        assert context.message == mock_params.message
        assert context.task_id == '00000000-0000-0000-0000-000000000001'
        assert (
            mock_params.message.taskId == '00000000-0000-0000-0000-000000000001'
        )
        assert context.context_id == '00000000-0000-0000-0000-000000000002'
        assert (
            mock_params.message.contextId
            == '00000000-0000-0000-0000-000000000002'
        )

    def test_init_with_task_id(self, mock_params):
        """Test initialization with task ID provided."""
        task_id = 'task-123'
        context = RequestContext(request=mock_params, task_id=task_id)

        assert context.task_id == task_id
        assert mock_params.message.taskId == task_id

    def test_init_with_context_id(self, mock_params):
        """Test initialization with context ID provided."""
        context_id = 'context-456'
        context = RequestContext(request=mock_params, context_id=context_id)

        assert context.context_id == context_id
        assert mock_params.message.contextId == context_id

    def test_init_with_both_ids(self, mock_params):
        """Test initialization with both task and context IDs provided."""
        task_id = 'task-123'
        context_id = 'context-456'
        context = RequestContext(
            request=mock_params, task_id=task_id, context_id=context_id
        )

        assert context.task_id == task_id
        assert mock_params.message.taskId == task_id
        assert context.context_id == context_id
        assert mock_params.message.contextId == context_id

    def test_init_with_task(self, mock_params, mock_task):
        """Test initialization with a task object."""
        context = RequestContext(request=mock_params, task=mock_task)

        assert context.current_task == mock_task

    def test_get_user_input_no_params(self):
        """Test get_user_input with no params returns empty string."""
        context = RequestContext()
        assert context.get_user_input() == ''

    def test_attach_related_task(self, mock_task):
        """Test attach_related_task adds a task to related_tasks."""
        context = RequestContext()
        assert len(context.related_tasks) == 0

        context.attach_related_task(mock_task)
        assert len(context.related_tasks) == 1
        assert context.related_tasks[0] == mock_task

        # Test adding multiple tasks
        another_task = Mock(spec=Task)
        context.attach_related_task(another_task)
        assert len(context.related_tasks) == 2
        assert context.related_tasks[1] == another_task

    def test_current_task_property(self, mock_task):
        """Test current_task getter and setter."""
        context = RequestContext()
        assert context.current_task is None

        context.current_task = mock_task
        assert context.current_task == mock_task

        # Change current task
        new_task = Mock(spec=Task)
        context.current_task = new_task
        assert context.current_task == new_task

    def test_check_or_generate_task_id_no_params(self):
        """Test _check_or_generate_task_id with no params does nothing."""
        context = RequestContext()
        context._check_or_generate_task_id()
        assert context.task_id is None

    def test_check_or_generate_task_id_with_existing_task_id(self, mock_params):
        """Test _check_or_generate_task_id with existing task ID."""
        existing_id = 'existing-task-id'
        mock_params.message.taskId = existing_id

        context = RequestContext(request=mock_params)
        # The method is called during initialization

        assert context.task_id == existing_id
        assert mock_params.message.taskId == existing_id

    def test_check_or_generate_context_id_no_params(self):
        """Test _check_or_generate_context_id with no params does nothing."""
        context = RequestContext()
        context._check_or_generate_context_id()
        assert context.context_id is None

    def test_check_or_generate_context_id_with_existing_context_id(
        self, mock_params
    ):
        """Test _check_or_generate_context_id with existing context ID."""
        existing_id = 'existing-context-id'
        mock_params.message.contextId = existing_id

        context = RequestContext(request=mock_params)
        # The method is called during initialization

        assert context.context_id == existing_id
        assert mock_params.message.contextId == existing_id

    def test_with_related_tasks_provided(self, mock_task):
        """Test initialization with related tasks provided."""
        related_tasks = [mock_task, Mock(spec=Task)]
        context = RequestContext(related_tasks=related_tasks)

        assert context.related_tasks == related_tasks
        assert len(context.related_tasks) == 2

    def test_message_property_without_params(self):
        """Test message property returns None when no params are provided."""
        context = RequestContext()
        assert context.message is None

    def test_message_property_with_params(self, mock_params):
        """Test message property returns the message from params."""
        context = RequestContext(request=mock_params)
        assert context.message == mock_params.message

    def test_init_with_existing_ids_in_message(self, mock_message, mock_params):
        """Test initialization with existing IDs in the message."""
        mock_message.taskId = 'existing-task-id'
        mock_message.contextId = 'existing-context-id'

        context = RequestContext(request=mock_params)

        assert context.task_id == 'existing-task-id'
        assert context.context_id == 'existing-context-id'
        # No new UUIDs should be generated

    def test_init_with_task_id_and_existing_task_id_match(
        self, mock_params, mock_task
    ):
        """Test initialization succeeds when task_id matches task.id."""
        mock_params.message.taskId = mock_task.id

        context = RequestContext(
            request=mock_params, task_id=mock_task.id, task=mock_task
        )

        assert context.task_id == mock_task.id
        assert context.current_task == mock_task

    def test_init_with_context_id_and_existing_context_id_match(
        self, mock_params, mock_task
    ):
        """Test initialization succeeds when context_id matches task.contextId."""
        mock_params.message.taskId = mock_task.id  # Set matching task ID
        mock_params.message.contextId = mock_task.contextId

        context = RequestContext(
            request=mock_params,
            task_id=mock_task.id,
            context_id=mock_task.contextId,
            task=mock_task,
        )

        assert context.context_id == mock_task.contextId
        assert context.current_task == mock_task
````

## File: tests/server/events/test_inmemory_queue_manager.py
````python
import asyncio

from unittest.mock import MagicMock

import pytest

from a2a.server.events import InMemoryQueueManager
from a2a.server.events.event_queue import EventQueue
from a2a.server.events.queue_manager import (
    NoTaskQueue,
    TaskQueueExists,
)


class TestInMemoryQueueManager:
    @pytest.fixture
    def queue_manager(self):
        """Fixture to create a fresh InMemoryQueueManager for each test."""
        manager = InMemoryQueueManager()
        return manager

    @pytest.fixture
    def event_queue(self):
        """Fixture to create a mock EventQueue."""
        queue = MagicMock(spec=EventQueue)
        # Mock the tap method to return itself
        queue.tap.return_value = queue
        return queue

    @pytest.mark.asyncio
    async def test_init(self, queue_manager):
        """Test that the InMemoryQueueManager initializes with empty task queue and a lock."""
        assert queue_manager._task_queue == {}
        assert isinstance(queue_manager._lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_add_new_queue(self, queue_manager, event_queue):
        """Test adding a new queue to the manager."""
        task_id = 'test_task_id'
        await queue_manager.add(task_id, event_queue)
        assert queue_manager._task_queue[task_id] == event_queue

    @pytest.mark.asyncio
    async def test_add_existing_queue(self, queue_manager, event_queue):
        """Test adding a queue with an existing task_id raises TaskQueueExists."""
        task_id = 'test_task_id'
        await queue_manager.add(task_id, event_queue)

        with pytest.raises(TaskQueueExists):
            await queue_manager.add(task_id, event_queue)

    @pytest.mark.asyncio
    async def test_get_existing_queue(self, queue_manager, event_queue):
        """Test getting an existing queue returns the queue."""
        task_id = 'test_task_id'
        await queue_manager.add(task_id, event_queue)

        result = await queue_manager.get(task_id)
        assert result == event_queue

    @pytest.mark.asyncio
    async def test_get_nonexistent_queue(self, queue_manager):
        """Test getting a nonexistent queue returns None."""
        result = await queue_manager.get('nonexistent_task_id')
        assert result is None

    @pytest.mark.asyncio
    async def test_tap_existing_queue(self, queue_manager, event_queue):
        """Test tapping an existing queue returns the tapped queue."""
        task_id = 'test_task_id'
        await queue_manager.add(task_id, event_queue)

        result = await queue_manager.tap(task_id)
        assert result == event_queue
        event_queue.tap.assert_called_once()

    @pytest.mark.asyncio
    async def test_tap_nonexistent_queue(self, queue_manager):
        """Test tapping a nonexistent queue returns None."""
        result = await queue_manager.tap('nonexistent_task_id')
        assert result is None

    @pytest.mark.asyncio
    async def test_close_existing_queue(self, queue_manager, event_queue):
        """Test closing an existing queue removes it from the manager."""
        task_id = 'test_task_id'
        await queue_manager.add(task_id, event_queue)

        await queue_manager.close(task_id)
        assert task_id not in queue_manager._task_queue

    @pytest.mark.asyncio
    async def test_close_nonexistent_queue(self, queue_manager):
        """Test closing a nonexistent queue raises NoTaskQueue."""
        with pytest.raises(NoTaskQueue):
            await queue_manager.close('nonexistent_task_id')

    @pytest.mark.asyncio
    async def test_create_or_tap_new_queue(self, queue_manager):
        """Test create_or_tap with a new task_id creates and returns a new queue."""
        task_id = 'test_task_id'

        result = await queue_manager.create_or_tap(task_id)
        assert isinstance(result, EventQueue)
        assert queue_manager._task_queue[task_id] == result

    @pytest.mark.asyncio
    async def test_create_or_tap_existing_queue(
        self, queue_manager, event_queue
    ):
        """Test create_or_tap with an existing task_id taps and returns the existing queue."""
        task_id = 'test_task_id'
        await queue_manager.add(task_id, event_queue)

        result = await queue_manager.create_or_tap(task_id)

        assert result == event_queue
        event_queue.tap.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrency(self, queue_manager):
        """Test concurrent access to the queue manager."""

        async def add_task(task_id):
            queue = EventQueue()
            await queue_manager.add(task_id, queue)
            return task_id

        async def get_task(task_id):
            return await queue_manager.get(task_id)

        # Create 10 different task IDs
        task_ids = [f'task_{i}' for i in range(10)]

        # Add tasks concurrently
        add_tasks = [add_task(task_id) for task_id in task_ids]
        added_task_ids = await asyncio.gather(*add_tasks)

        # Verify all tasks were added
        assert set(added_task_ids) == set(task_ids)

        # Get tasks concurrently
        get_tasks = [get_task(task_id) for task_id in task_ids]
        queues = await asyncio.gather(*get_tasks)

        # Verify all queues are not None
        assert all(queue is not None for queue in queues)

        # Verify all tasks are in the manager
        for task_id in task_ids:
            assert task_id in queue_manager._task_queue
````

## File: tests/utils/test_message.py
````python
import uuid

from unittest.mock import patch

from a2a.types import (
    Message,
    Part,
    Role,
    TextPart,
)
from a2a.utils import get_message_text, get_text_parts, new_agent_text_message


class TestNewAgentTextMessage:
    def test_new_agent_text_message_basic(self):
        # Setup
        text = "Hello, I'm an agent"

        # Exercise - with a fixed uuid for testing
        with patch(
            'uuid.uuid4',
            return_value=uuid.UUID('12345678-1234-5678-1234-567812345678'),
        ):
            message = new_agent_text_message(text)

        # Verify
        assert message.role == Role.agent
        assert len(message.parts) == 1
        assert message.parts[0].root.text == text
        assert message.messageId == '12345678-1234-5678-1234-567812345678'
        assert message.taskId is None
        assert message.contextId is None

    def test_new_agent_text_message_with_context_id(self):
        # Setup
        text = 'Message with context'
        context_id = 'test-context-id'

        # Exercise
        with patch(
            'uuid.uuid4',
            return_value=uuid.UUID('12345678-1234-5678-1234-567812345678'),
        ):
            message = new_agent_text_message(text, context_id=context_id)

        # Verify
        assert message.role == Role.agent
        assert message.parts[0].root.text == text
        assert message.messageId == '12345678-1234-5678-1234-567812345678'
        assert message.contextId == context_id
        assert message.taskId is None

    def test_new_agent_text_message_with_task_id(self):
        # Setup
        text = 'Message with task id'
        task_id = 'test-task-id'

        # Exercise
        with patch(
            'uuid.uuid4',
            return_value=uuid.UUID('12345678-1234-5678-1234-567812345678'),
        ):
            message = new_agent_text_message(text, task_id=task_id)

        # Verify
        assert message.role == Role.agent
        assert message.parts[0].root.text == text
        assert message.messageId == '12345678-1234-5678-1234-567812345678'
        assert message.taskId == task_id
        assert message.contextId is None

    def test_new_agent_text_message_with_both_ids(self):
        # Setup
        text = 'Message with both ids'
        context_id = 'test-context-id'
        task_id = 'test-task-id'

        # Exercise
        with patch(
            'uuid.uuid4',
            return_value=uuid.UUID('12345678-1234-5678-1234-567812345678'),
        ):
            message = new_agent_text_message(
                text, context_id=context_id, task_id=task_id
            )

        # Verify
        assert message.role == Role.agent
        assert message.parts[0].root.text == text
        assert message.messageId == '12345678-1234-5678-1234-567812345678'
        assert message.contextId == context_id
        assert message.taskId == task_id

    def test_new_agent_text_message_empty_text(self):
        # Setup
        text = ''

        # Exercise
        with patch(
            'uuid.uuid4',
            return_value=uuid.UUID('12345678-1234-5678-1234-567812345678'),
        ):
            message = new_agent_text_message(text)

        # Verify
        assert message.role == Role.agent
        assert message.parts[0].root.text == ''
        assert message.messageId == '12345678-1234-5678-1234-567812345678'


class TestGetTextParts:
    def test_get_text_parts_single_text_part(self):
        # Setup
        parts = [Part(root=TextPart(text='Hello world'))]

        # Exercise
        result = get_text_parts(parts)

        # Verify
        assert result == ['Hello world']

    def test_get_text_parts_multiple_text_parts(self):
        # Setup
        parts = [
            Part(root=TextPart(text='First part')),
            Part(root=TextPart(text='Second part')),
            Part(root=TextPart(text='Third part')),
        ]

        # Exercise
        result = get_text_parts(parts)

        # Verify
        assert result == ['First part', 'Second part', 'Third part']

    def test_get_text_parts_empty_list(self):
        # Setup
        parts = []

        # Exercise
        result = get_text_parts(parts)

        # Verify
        assert result == []


class TestGetMessageText:
    def test_get_message_text_single_part(self):
        # Setup
        message = Message(
            role=Role.agent,
            parts=[Part(root=TextPart(text='Hello world'))],
            messageId='test-message-id',
        )

        # Exercise
        result = get_message_text(message)

        # Verify
        assert result == 'Hello world'

    def test_get_message_text_multiple_parts(self):
        # Setup
        message = Message(
            role=Role.agent,
            parts=[
                Part(root=TextPart(text='First line')),
                Part(root=TextPart(text='Second line')),
                Part(root=TextPart(text='Third line')),
            ],
            messageId='test-message-id',
        )

        # Exercise
        result = get_message_text(message)

        # Verify - default delimiter is newline
        assert result == 'First line\nSecond line\nThird line'

    def test_get_message_text_custom_delimiter(self):
        # Setup
        message = Message(
            role=Role.agent,
            parts=[
                Part(root=TextPart(text='First part')),
                Part(root=TextPart(text='Second part')),
                Part(root=TextPart(text='Third part')),
            ],
            messageId='test-message-id',
        )

        # Exercise
        result = get_message_text(message, delimiter=' | ')

        # Verify
        assert result == 'First part | Second part | Third part'

    def test_get_message_text_empty_parts(self):
        # Setup
        message = Message(
            role=Role.agent,
            parts=[],
            messageId='test-message-id',
        )

        # Exercise
        result = get_message_text(message)

        # Verify
        assert result == ''
````

## File: tests/README.md
````markdown
## Running the tests

1. Run the tests
    ```bash
    uv run pytest -v -s client/test_client.py
    ```

In case of failures, you can cleanup the cache:

1. `uv clean`
2. `rm -fR .pytest_cache .venv __pycache__`
````

## File: .github/PULL_REQUEST_TEMPLATE.md
````markdown
# Description

Thank you for opening a Pull Request!
Before submitting your PR, there are a few things you can do to make sure it goes smoothly:

- [ ] Follow the [`CONTRIBUTING` Guide](https://github.com/google-a2a/a2a-python/blob/main/CONTRIBUTING.md).
- [ ] Make your Pull Request title in the <https://www.conventionalcommits.org/> specification.
- [ ] Ensure the tests and linter pass (Run `nox -s format` from the repository root to format)
- [ ] Appropriate docs were updated (if necessary)

Fixes #<issue_number_goes_here> 🦕
````

## File: src/a2a/client/__init__.py
````python
"""Client-side components for interacting with an A2A agent."""

from a2a.client.client import A2ACardResolver, A2AClient
from a2a.client.errors import (
    A2AClientError,
    A2AClientHTTPError,
    A2AClientJSONError,
)
from a2a.client.helpers import create_text_message_object


__all__ = [
    'A2ACardResolver',
    'A2AClient',
    'A2AClientError',
    'A2AClientHTTPError',
    'A2AClientJSONError',
    'create_text_message_object',
]
````

## File: src/a2a/client/helpers.py
````python
"""Helper functions for the A2A client."""

from uuid import uuid4

from a2a.types import Message, Part, Role, TextPart


def create_text_message_object(
    role: Role = Role.user, content: str = ''
) -> Message:
    """Create a Message object containing a single TextPart.

    Args:
        role: The role of the message sender (user or agent). Defaults to Role.user.
        content: The text content of the message. Defaults to an empty string.

    Returns:
        A `Message` object with a new UUID messageId.
    """
    return Message(
        role=role, parts=[Part(TextPart(text=content))], messageId=str(uuid4())
    )
````

## File: src/a2a/server/agent_execution/agent_executor.py
````python
from abc import ABC, abstractmethod

from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue


class AgentExecutor(ABC):
    """Agent Executor interface.

    Implementations of this interface contain the core logic of the agent,
    executing tasks based on requests and publishing updates to an event queue.
    """

    @abstractmethod
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Execute the agent's logic for a given request context.

        The agent should read necessary information from the `context` and
        publish `Task` or `Message` events, or `TaskStatusUpdateEvent` /
        `TaskArtifactUpdateEvent` to the `event_queue`. This method should
        return once the agent's execution for this request is complete or
        yields control (e.g., enters an input-required state).

        Args:
            context: The request context containing the message, task ID, etc.
            event_queue: The queue to publish events to.
        """

    @abstractmethod
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Request the agent to cancel an ongoing task.

        The agent should attempt to stop the task identified by the task_id
        in the context and publish a `TaskStatusUpdateEvent` with state
        `TaskState.canceled` to the `event_queue`.

        Args:
            context: The request context containing the task ID to cancel.
            event_queue: The queue to publish the cancellation status update to.
        """
````

## File: src/a2a/server/agent_execution/request_context_builder.py
````python
from abc import ABC, abstractmethod

from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.types import MessageSendParams, Task


class RequestContextBuilder(ABC):
    """Builds request context to be supplied to agent executor"""

    @abstractmethod
    async def build(
        self,
        params: MessageSendParams | None = None,
        task_id: str | None = None,
        context_id: str | None = None,
        task: Task | None = None,
        context: ServerCallContext | None = None,
    ) -> RequestContext:
        pass
````

## File: src/a2a/server/agent_execution/simple_request_context_builder.py
````python
import asyncio

from a2a.server.agent_execution import RequestContext, RequestContextBuilder
from a2a.server.context import ServerCallContext
from a2a.server.tasks import TaskStore
from a2a.types import MessageSendParams, Task


class SimpleRequestContextBuilder(RequestContextBuilder):
    """Builds request context and populates referred tasks"""

    def __init__(
        self,
        should_populate_referred_tasks: bool = False,
        task_store: TaskStore | None = None,
    ) -> None:
        self._task_store = task_store
        self._should_populate_referred_tasks = should_populate_referred_tasks

    async def build(
        self,
        params: MessageSendParams | None = None,
        task_id: str | None = None,
        context_id: str | None = None,
        task: Task | None = None,
        context: ServerCallContext | None = None,
    ) -> RequestContext:
        related_tasks: list[Task] | None = None

        if (
            self._task_store
            and self._should_populate_referred_tasks
            and params
            and params.message.referenceTaskIds
        ):
            tasks = await asyncio.gather(
                *[
                    self._task_store.get(task_id)
                    for task_id in params.message.referenceTaskIds
                ]
            )
            related_tasks = [x for x in tasks if x is not None]

        return RequestContext(
            request=params,
            task_id=task_id,
            context_id=context_id,
            task=task,
            related_tasks=related_tasks,
            call_context=context,
        )
````

## File: src/a2a/server/request_handlers/__init__.py
````python
"""Request handler components for the A2A server."""

from a2a.server.request_handlers.default_request_handler import (
    DefaultRequestHandler,
)
from a2a.server.request_handlers.jsonrpc_handler import JSONRPCHandler
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.request_handlers.response_helpers import (
    build_error_response,
    prepare_response_object,
)


__all__ = [
    'DefaultRequestHandler',
    'JSONRPCHandler',
    'RequestHandler',
    'build_error_response',
    'prepare_response_object',
]
````

## File: src/a2a/server/request_handlers/response_helpers.py
````python
"""Helper functions for building A2A JSON-RPC responses."""

# response types
from typing import TypeVar

from a2a.types import (
    A2AError,
    CancelTaskResponse,
    CancelTaskSuccessResponse,
    GetTaskPushNotificationConfigResponse,
    GetTaskPushNotificationConfigSuccessResponse,
    GetTaskResponse,
    GetTaskSuccessResponse,
    InvalidAgentResponseError,
    JSONRPCError,
    JSONRPCErrorResponse,
    Message,
    SendMessageResponse,
    SendMessageSuccessResponse,
    SendStreamingMessageResponse,
    SendStreamingMessageSuccessResponse,
    SetTaskPushNotificationConfigResponse,
    SetTaskPushNotificationConfigSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskPushNotificationConfig,
    TaskStatusUpdateEvent,
)


RT = TypeVar(
    'RT',
    GetTaskResponse,
    CancelTaskResponse,
    SendMessageResponse,
    SetTaskPushNotificationConfigResponse,
    GetTaskPushNotificationConfigResponse,
    SendStreamingMessageResponse,
)
"""Type variable for RootModel response types."""

# success types
SPT = TypeVar(
    'SPT',
    GetTaskSuccessResponse,
    CancelTaskSuccessResponse,
    SendMessageSuccessResponse,
    SetTaskPushNotificationConfigSuccessResponse,
    GetTaskPushNotificationConfigSuccessResponse,
    SendStreamingMessageSuccessResponse,
)
"""Type variable for SuccessResponse types."""

# result types
EventTypes = (
    Task
    | Message
    | TaskArtifactUpdateEvent
    | TaskStatusUpdateEvent
    | TaskPushNotificationConfig
    | A2AError
    | JSONRPCError
)
"""Type alias for possible event types produced by handlers."""


def build_error_response(
    request_id: str | int | None,
    error: A2AError | JSONRPCError,
    response_wrapper_type: type[RT],
) -> RT:
    """Helper method to build a JSONRPCErrorResponse wrapped in the appropriate response type.

    Args:
        request_id: The ID of the request that caused the error.
        error: The A2AError or JSONRPCError object.
        response_wrapper_type: The Pydantic RootModel type that wraps the response
                                for the specific RPC method (e.g., `SendMessageResponse`).

    Returns:
        A Pydantic model representing the JSON-RPC error response,
        wrapped in the specified response type.
    """
    return response_wrapper_type(
        JSONRPCErrorResponse(
            id=request_id,
            error=error.root if isinstance(error, A2AError) else error,
        )
    )


def prepare_response_object(
    request_id: str | int | None,
    response: EventTypes,
    success_response_types: tuple[type, ...],
    success_payload_type: type[SPT],
    response_type: type[RT],
) -> RT:
    """Helper method to build appropriate JSONRPCResponse object for RPC methods.

    Based on the type of the `response` object received from the handler,
    it constructs either a success response wrapped in the appropriate payload type
    or an error response.

    Args:
        request_id: The ID of the request.
        response: The object received from the request handler.
        success_response_types: A tuple of expected Pydantic model types for a successful result.
        success_payload_type: The Pydantic model type for the success payload
                                (e.g., `SendMessageSuccessResponse`).
        response_type: The Pydantic RootModel type that wraps the final response
                       (e.g., `SendMessageResponse`).

    Returns:
        A Pydantic model representing the final JSON-RPC response (success or error).
    """
    if isinstance(response, success_response_types):
        return response_type(
            root=success_payload_type(id=request_id, result=response)  # type:ignore
        )

    if isinstance(response, A2AError | JSONRPCError):
        return build_error_response(request_id, response, response_type)

    # If consumer_data is not an expected success type and not an error,
    # it's an invalid type of response from the agent for this specific method.
    response = A2AError(
        root=InvalidAgentResponseError(
            message='Agent returned invalid type response for this method'
        )
    )

    return build_error_response(request_id, response, response_type)
````

## File: src/a2a/server/tasks/inmemory_push_notifier.py
````python
import asyncio
import logging

import httpx

from a2a.server.tasks.push_notifier import PushNotifier
from a2a.types import PushNotificationConfig, Task


logger = logging.getLogger(__name__)


class InMemoryPushNotifier(PushNotifier):
    """In-memory implementation of PushNotifier interface.

    Stores push notification configurations in memory and uses an httpx client
    to send notifications.
    """

    def __init__(self, httpx_client: httpx.AsyncClient) -> None:
        """Initializes the InMemoryPushNotifier.

        Args:
            httpx_client: An async HTTP client instance to send notifications.
        """
        self._client = httpx_client
        self.lock = asyncio.Lock()
        self._push_notification_infos: dict[str, PushNotificationConfig] = {}

    async def set_info(
        self, task_id: str, notification_config: PushNotificationConfig
    ):
        """Sets or updates the push notification configuration for a task in memory."""
        async with self.lock:
            self._push_notification_infos[task_id] = notification_config

    async def get_info(self, task_id: str) -> PushNotificationConfig | None:
        """Retrieves the push notification configuration for a task from memory."""
        async with self.lock:
            return self._push_notification_infos.get(task_id)

    async def delete_info(self, task_id: str):
        """Deletes the push notification configuration for a task from memory."""
        async with self.lock:
            if task_id in self._push_notification_infos:
                del self._push_notification_infos[task_id]

    async def send_notification(self, task: Task):
        """Sends a push notification for a task if configuration exists."""
        push_info = await self.get_info(task.id)
        if not push_info:
            return
        url = push_info.url

        try:
            response = await self._client.post(
                url, json=task.model_dump(mode='json', exclude_none=True)
            )
            response.raise_for_status()
            logger.info(f'Push-notification sent for URL: {url}')
        except Exception as e:
            logger.error(f'Error sending push-notification: {e}')
````

## File: src/a2a/server/tasks/push_notifier.py
````python
from abc import ABC, abstractmethod

from a2a.types import PushNotificationConfig, Task


class PushNotifier(ABC):
    """PushNotifier interface to store, retrieve push notification for tasks and send push notifications."""

    @abstractmethod
    async def set_info(
        self, task_id: str, notification_config: PushNotificationConfig
    ):
        """Sets or updates the push notification configuration for a task."""

    @abstractmethod
    async def get_info(self, task_id: str) -> PushNotificationConfig | None:
        """Retrieves the push notification configuration for a task."""

    @abstractmethod
    async def delete_info(self, task_id: str):
        """Deletes the push notification configuration for a task."""

    @abstractmethod
    async def send_notification(self, task: Task):
        """Sends a push notification containing the latest task state."""
````

## File: src/a2a/server/context.py
````python
"""Defines the ServerCallContext class."""

import collections.abc
import typing

from pydantic import BaseModel, ConfigDict, Field

from a2a.auth.user import UnauthenticatedUser, User


State = collections.abc.MutableMapping[str, typing.Any]


class ServerCallContext(BaseModel):
    """A context passed when calling a server method.

    This class allows storing arbitrary user data in the state attribute.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: State = Field(default={})
    user: User = Field(default=UnauthenticatedUser())
````

## File: tests/server/events/test_event_queue.py
````python
import asyncio
import pytest
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    A2AError,
    JSONRPCError,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TaskStatus,
    TaskState,
    Artifact,
    Part,
    TextPart,
    TaskNotFoundError,
)
from typing import Any

MINIMAL_TASK: dict[str, Any] = {
    'id': '123',
    'contextId': 'session-xyz',
    'status': {'state': 'submitted'},
    'kind': 'task',
}
MESSAGE_PAYLOAD: dict[str, Any] = {
    'role': 'agent',
    'parts': [{'text': 'test message'}],
    'messageId': '111',
}


@pytest.fixture
def event_queue() -> EventQueue:
    return EventQueue()


@pytest.mark.asyncio
async def test_enqueue_and_dequeue_event(event_queue: EventQueue) -> None:
    """Test that an event can be enqueued and dequeued."""
    event = Message(**MESSAGE_PAYLOAD)
    event_queue.enqueue_event(event)
    dequeued_event = await event_queue.dequeue_event()
    assert dequeued_event == event


@pytest.mark.asyncio
async def test_dequeue_event_no_wait(event_queue: EventQueue) -> None:
    """Test dequeue_event with no_wait=True."""
    event = Task(**MINIMAL_TASK)
    event_queue.enqueue_event(event)
    dequeued_event = await event_queue.dequeue_event(no_wait=True)
    assert dequeued_event == event


@pytest.mark.asyncio
async def test_dequeue_event_empty_queue_no_wait(
    event_queue: EventQueue,
) -> None:
    """Test dequeue_event with no_wait=True when the queue is empty."""
    with pytest.raises(asyncio.QueueEmpty):
        await event_queue.dequeue_event(no_wait=True)


@pytest.mark.asyncio
async def test_dequeue_event_wait(event_queue: EventQueue) -> None:
    """Test dequeue_event with the default wait behavior."""
    event = TaskStatusUpdateEvent(
        taskId='task_123',
        contextId='session-xyz',
        status=TaskStatus(state=TaskState.working),
        final=True,
    )
    event_queue.enqueue_event(event)
    dequeued_event = await event_queue.dequeue_event()
    assert dequeued_event == event


@pytest.mark.asyncio
async def test_task_done(event_queue: EventQueue) -> None:
    """Test the task_done method."""
    event = TaskArtifactUpdateEvent(
        taskId='task_123',
        contextId='session-xyz',
        artifact=Artifact(artifactId='11', parts=[Part(TextPart(text='text'))]),
    )
    event_queue.enqueue_event(event)
    _ = await event_queue.dequeue_event()
    event_queue.task_done()


@pytest.mark.asyncio
async def test_enqueue_different_event_types(
    event_queue: EventQueue,
) -> None:
    """Test enqueuing different types of events."""
    events: list[Any] = [
        A2AError(TaskNotFoundError()),
        JSONRPCError(code=111, message='rpc error'),
    ]
    for event in events:
        event_queue.enqueue_event(event)
        dequeued_event = await event_queue.dequeue_event()
        assert dequeued_event == event
````

## File: tests/server/tasks/test_task_updater.py
````python
import uuid

from unittest.mock import Mock, patch

import pytest

from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)


class TestTaskUpdater:
    @pytest.fixture
    def event_queue(self):
        """Create a mock event queue for testing."""
        return Mock(spec=EventQueue)

    @pytest.fixture
    def task_updater(self, event_queue):
        """Create a TaskUpdater instance for testing."""
        return TaskUpdater(
            event_queue=event_queue,
            task_id='test-task-id',
            context_id='test-context-id',
        )

    @pytest.fixture
    def sample_message(self):
        """Create a sample message for testing."""
        return Message(
            role=Role.agent,
            taskId='test-task-id',
            contextId='test-context-id',
            messageId='test-message-id',
            parts=[Part(root=TextPart(text='Test message'))],
        )

    @pytest.fixture
    def sample_parts(self):
        """Create sample parts for testing."""
        return [Part(root=TextPart(text='Test part'))]

    def test_init(self, event_queue):
        """Test that TaskUpdater initializes correctly."""
        task_updater = TaskUpdater(
            event_queue=event_queue,
            task_id='test-task-id',
            context_id='test-context-id',
        )

        assert task_updater.event_queue == event_queue
        assert task_updater.task_id == 'test-task-id'
        assert task_updater.context_id == 'test-context-id'

    def test_update_status_without_message(self, task_updater, event_queue):
        """Test updating status without a message."""
        task_updater.update_status(TaskState.working)

        event_queue.enqueue_event.assert_called_once()
        event = event_queue.enqueue_event.call_args[0][0]

        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.taskId == 'test-task-id'
        assert event.contextId == 'test-context-id'
        assert event.final is False
        assert event.status.state == TaskState.working
        assert event.status.message is None

    def test_update_status_with_message(
        self, task_updater, event_queue, sample_message
    ):
        """Test updating status with a message."""
        task_updater.update_status(TaskState.working, message=sample_message)

        event_queue.enqueue_event.assert_called_once()
        event = event_queue.enqueue_event.call_args[0][0]

        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.taskId == 'test-task-id'
        assert event.contextId == 'test-context-id'
        assert event.final is False
        assert event.status.state == TaskState.working
        assert event.status.message == sample_message

    def test_update_status_final(self, task_updater, event_queue):
        """Test updating status with final=True."""
        task_updater.update_status(TaskState.completed, final=True)

        event_queue.enqueue_event.assert_called_once()
        event = event_queue.enqueue_event.call_args[0][0]

        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.final is True
        assert event.status.state == TaskState.completed

    def test_add_artifact_with_custom_id_and_name(
        self, task_updater, event_queue, sample_parts
    ):
        """Test adding an artifact with a custom ID and name."""
        task_updater.add_artifact(
            parts=sample_parts,
            artifact_id='custom-artifact-id',
            name='Custom Artifact',
        )

        event_queue.enqueue_event.assert_called_once()
        event = event_queue.enqueue_event.call_args[0][0]

        assert isinstance(event, TaskArtifactUpdateEvent)
        assert event.artifact.artifactId == 'custom-artifact-id'
        assert event.artifact.name == 'Custom Artifact'
        assert event.artifact.parts == sample_parts

    def test_complete_without_message(self, task_updater, event_queue):
        """Test marking a task as completed without a message."""
        task_updater.complete()

        event_queue.enqueue_event.assert_called_once()
        event = event_queue.enqueue_event.call_args[0][0]

        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.status.state == TaskState.completed
        assert event.final is True
        assert event.status.message is None

    def test_complete_with_message(
        self, task_updater, event_queue, sample_message
    ):
        """Test marking a task as completed with a message."""
        task_updater.complete(message=sample_message)

        event_queue.enqueue_event.assert_called_once()
        event = event_queue.enqueue_event.call_args[0][0]

        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.status.state == TaskState.completed
        assert event.final is True
        assert event.status.message == sample_message

    def test_submit_without_message(self, task_updater, event_queue):
        """Test marking a task as submitted without a message."""
        task_updater.submit()

        event_queue.enqueue_event.assert_called_once()
        event = event_queue.enqueue_event.call_args[0][0]

        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.status.state == TaskState.submitted
        assert event.final is False
        assert event.status.message is None

    def test_submit_with_message(
        self, task_updater, event_queue, sample_message
    ):
        """Test marking a task as submitted with a message."""
        task_updater.submit(message=sample_message)

        event_queue.enqueue_event.assert_called_once()
        event = event_queue.enqueue_event.call_args[0][0]

        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.status.state == TaskState.submitted
        assert event.final is False
        assert event.status.message == sample_message

    def test_start_work_without_message(self, task_updater, event_queue):
        """Test marking a task as working without a message."""
        task_updater.start_work()

        event_queue.enqueue_event.assert_called_once()
        event = event_queue.enqueue_event.call_args[0][0]

        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.status.state == TaskState.working
        assert event.final is False
        assert event.status.message is None

    def test_start_work_with_message(
        self, task_updater, event_queue, sample_message
    ):
        """Test marking a task as working with a message."""
        task_updater.start_work(message=sample_message)

        event_queue.enqueue_event.assert_called_once()
        event = event_queue.enqueue_event.call_args[0][0]

        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.status.state == TaskState.working
        assert event.final is False
        assert event.status.message == sample_message

    def test_new_agent_message(self, task_updater, sample_parts):
        """Test creating a new agent message."""
        with patch(
            'uuid.uuid4',
            return_value=uuid.UUID('12345678-1234-5678-1234-567812345678'),
        ):
            message = task_updater.new_agent_message(parts=sample_parts)

        assert message.role == Role.agent
        assert message.taskId == 'test-task-id'
        assert message.contextId == 'test-context-id'
        assert message.messageId == '12345678-1234-5678-1234-567812345678'
        assert message.parts == sample_parts
        assert message.metadata is None

    def test_new_agent_message_with_metadata(
        self, task_updater, sample_parts
    ):
        """Test creating a new agent message with metadata and final=True."""
        metadata = {'key': 'value'}

        with patch(
            'uuid.uuid4',
            return_value=uuid.UUID('12345678-1234-5678-1234-567812345678'),
        ):
            message = task_updater.new_agent_message(
                parts=sample_parts, metadata=metadata
            )

        assert message.role == Role.agent
        assert message.taskId == 'test-task-id'
        assert message.contextId == 'test-context-id'
        assert message.messageId == '12345678-1234-5678-1234-567812345678'
        assert message.parts == sample_parts
        assert message.metadata == metadata
````

## File: tests/utils/test_helpers.py
````python
from typing import Any

import pytest

from a2a.types import (
    Artifact,
    Message,
    MessageSendParams,
    Part,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TextPart,
)
from a2a.utils.errors import ServerError
from a2a.utils.helpers import (
    append_artifact_to_task,
    build_text_artifact,
    create_task_obj,
    validate,
)


# --- Helper Data ---
TEXT_PART_DATA: dict[str, Any] = {'type': 'text', 'text': 'Hello'}

MINIMAL_MESSAGE_USER: dict[str, Any] = {
    'role': 'user',
    'parts': [TEXT_PART_DATA],
    'messageId': 'msg-123',
    'type': 'message',
}

MINIMAL_TASK_STATUS: dict[str, Any] = {'state': 'submitted'}

MINIMAL_TASK: dict[str, Any] = {
    'id': 'task-abc',
    'contextId': 'session-xyz',
    'status': MINIMAL_TASK_STATUS,
    'type': 'task',
}


# Test create_task_obj
def test_create_task_obj():
    message = Message(**MINIMAL_MESSAGE_USER)
    send_params = MessageSendParams(message=message)

    task = create_task_obj(send_params)
    assert task.id is not None
    assert task.contextId == message.contextId
    assert task.status.state == TaskState.submitted
    assert len(task.history) == 1
    assert task.history[0] == message


# Test append_artifact_to_task
def test_append_artifact_to_task():
    # Prepare base task
    task = Task(**MINIMAL_TASK)
    assert task.id == 'task-abc'
    assert task.contextId == 'session-xyz'
    assert task.status.state == TaskState.submitted
    assert task.history is None
    assert task.artifacts is None
    assert task.metadata is None

    # Prepare appending artifact and event
    artifact_1 = Artifact(
        artifactId='artifact-123', parts=[Part(root=TextPart(text='Hello'))]
    )
    append_event_1 = TaskArtifactUpdateEvent(
        artifact=artifact_1, append=False, taskId='123', contextId='123'
    )

    # Test adding a new artifact (not appending)
    append_artifact_to_task(task, append_event_1)
    assert len(task.artifacts) == 1
    assert task.artifacts[0].artifactId == 'artifact-123'
    assert task.artifacts[0].name is None
    assert len(task.artifacts[0].parts) == 1
    assert task.artifacts[0].parts[0].root.text == 'Hello'

    # Test replacing the artifact
    artifact_2 = Artifact(
        artifactId='artifact-123',
        name='updated name',
        parts=[Part(root=TextPart(text='Updated'))],
    )
    append_event_2 = TaskArtifactUpdateEvent(
        artifact=artifact_2, append=False, taskId='123', contextId='123'
    )
    append_artifact_to_task(task, append_event_2)
    assert len(task.artifacts) == 1  # Should still have one artifact
    assert task.artifacts[0].artifactId == 'artifact-123'
    assert task.artifacts[0].name == 'updated name'
    assert len(task.artifacts[0].parts) == 1
    assert task.artifacts[0].parts[0].root.text == 'Updated'

    # Test appending parts to an existing artifact
    artifact_with_parts = Artifact(
        artifactId='artifact-123', parts=[Part(root=TextPart(text='Part 2'))]
    )
    append_event_3 = TaskArtifactUpdateEvent(
        artifact=artifact_with_parts, append=True, taskId='123', contextId='123'
    )
    append_artifact_to_task(task, append_event_3)
    assert len(task.artifacts[0].parts) == 2
    assert task.artifacts[0].parts[0].root.text == 'Updated'
    assert task.artifacts[0].parts[1].root.text == 'Part 2'

    # Test adding another new artifact
    another_artifact_with_parts = Artifact(
        artifactId='new_artifact',
        parts=[Part(root=TextPart(text='new artifact Part 1'))],
    )
    append_event_4 = TaskArtifactUpdateEvent(
        artifact=another_artifact_with_parts,
        append=False,
        taskId='123',
        contextId='123',
    )
    append_artifact_to_task(task, append_event_4)
    assert len(task.artifacts) == 2
    assert task.artifacts[0].artifactId == 'artifact-123'
    assert task.artifacts[1].artifactId == 'new_artifact'
    assert len(task.artifacts[0].parts) == 2
    assert len(task.artifacts[1].parts) == 1

    # Test appending part to a task that does not have a matching artifact
    non_existing_artifact_with_parts = Artifact(
        artifactId='artifact-456', parts=[Part(root=TextPart(text='Part 1'))]
    )
    append_event_5 = TaskArtifactUpdateEvent(
        artifact=non_existing_artifact_with_parts,
        append=True,
        taskId='123',
        contextId='123',
    )
    append_artifact_to_task(task, append_event_5)
    assert len(task.artifacts) == 2
    assert len(task.artifacts[0].parts) == 2
    assert len(task.artifacts[1].parts) == 1


# Test build_text_artifact
def test_build_text_artifact():
    artifact_id = 'text_artifact'
    text = 'This is a sample text'
    artifact = build_text_artifact(text, artifact_id)

    assert artifact.artifactId == artifact_id
    assert len(artifact.parts) == 1
    assert artifact.parts[0].root.text == text


# Test validate decorator
def test_validate_decorator():
    class TestClass:
        condition = True

        @validate(lambda self: self.condition, 'Condition not met')
        def test_method(self):
            return 'Success'

    obj = TestClass()

    # Test passing condition
    assert obj.test_method() == 'Success'

    # Test failing condition
    obj.condition = False
    with pytest.raises(ServerError) as exc_info:
        obj.test_method()
    assert 'Condition not met' in str(exc_info.value)
````

## File: tests/utils/test_telemetry.py
````python
import asyncio

from unittest import mock

import pytest

from a2a.utils.telemetry import trace_class, trace_function


@pytest.fixture
def mock_span():
    return mock.MagicMock()


@pytest.fixture
def mock_tracer(mock_span):
    tracer = mock.MagicMock()
    tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    tracer.start_as_current_span.return_value.__exit__.return_value = False
    return tracer


@pytest.fixture(autouse=True)
def patch_trace_get_tracer(mock_tracer):
    with mock.patch('opentelemetry.trace.get_tracer', return_value=mock_tracer):
        yield


def test_trace_function_sync_success(mock_span):
    @trace_function
    def foo(x, y):
        return x + y

    result = foo(2, 3)
    assert result == 5
    mock_span.set_status.assert_called()
    mock_span.set_status.assert_any_call(mock.ANY)
    mock_span.record_exception.assert_not_called()


def test_trace_function_sync_exception(mock_span):
    @trace_function
    def bar():
        raise ValueError('fail')

    with pytest.raises(ValueError):
        bar()
    mock_span.record_exception.assert_called()
    mock_span.set_status.assert_any_call(mock.ANY, description='fail')


def test_trace_function_sync_attribute_extractor_called(mock_span):
    called = {}

    def attr_extractor(span, args, kwargs, result, exception):
        called['called'] = True
        assert span is mock_span
        assert exception is None
        assert result == 42

    @trace_function(attribute_extractor=attr_extractor)
    def foo():
        return 42

    foo()
    assert called['called']


def test_trace_function_sync_attribute_extractor_error_logged(mock_span):
    with mock.patch('a2a.utils.telemetry.logger') as logger:

        def attr_extractor(span, args, kwargs, result, exception):
            raise RuntimeError('attr fail')

        @trace_function(attribute_extractor=attr_extractor)
        def foo():
            return 1

        foo()
        logger.error.assert_any_call(mock.ANY)


@pytest.mark.asyncio
async def test_trace_function_async_success(mock_span):
    @trace_function
    async def foo(x):
        await asyncio.sleep(0)
        return x * 2

    result = await foo(4)
    assert result == 8
    mock_span.set_status.assert_called()
    mock_span.record_exception.assert_not_called()


@pytest.mark.asyncio
async def test_trace_function_async_exception(mock_span):
    @trace_function
    async def bar():
        await asyncio.sleep(0)
        raise RuntimeError('async fail')

    with pytest.raises(RuntimeError):
        await bar()
    mock_span.record_exception.assert_called()
    mock_span.set_status.assert_any_call(mock.ANY, description='async fail')


@pytest.mark.asyncio
async def test_trace_function_async_attribute_extractor_called(mock_span):
    called = {}

    def attr_extractor(span, args, kwargs, result, exception):
        called['called'] = True
        assert exception is None
        assert result == 99

    @trace_function(attribute_extractor=attr_extractor)
    async def foo():
        return 99

    await foo()
    assert called['called']


def test_trace_function_with_args_and_attributes(mock_span):
    @trace_function(span_name='custom.span', attributes={'foo': 'bar'})
    def foo():
        return 1

    foo()
    mock_span.set_attribute.assert_any_call('foo', 'bar')


def test_trace_class_exclude_list(mock_span):
    @trace_class(exclude_list=['skip_me'])
    class MyClass:
        def a(self):
            return 'a'

        def skip_me(self):
            return 'skip'

        def __str__(self):
            return 'str'

    obj = MyClass()
    assert obj.a() == 'a'
    assert obj.skip_me() == 'skip'
    # Only 'a' is traced, not 'skip_me' or dunder
    assert hasattr(obj.a, '__wrapped__')
    assert not hasattr(obj.skip_me, '__wrapped__')


def test_trace_class_include_list(mock_span):
    @trace_class(include_list=['only_this'])
    class MyClass:
        def only_this(self):
            return 'yes'

        def not_this(self):
            return 'no'

    obj = MyClass()
    assert obj.only_this() == 'yes'
    assert obj.not_this() == 'no'
    assert hasattr(obj.only_this, '__wrapped__')
    assert not hasattr(obj.not_this, '__wrapped__')


def test_trace_class_dunder_not_traced(mock_span):
    @trace_class()
    class MyClass:
        def __init__(self):
            self.x = 1

        def foo(self):
            return 'foo'

    obj = MyClass()
    assert obj.foo() == 'foo'
    assert hasattr(obj.foo, '__wrapped__')
    assert hasattr(obj, 'x')
````

## File: tests/test_types.py
````python
from typing import Any

import pytest

from pydantic import ValidationError

from a2a.types import (
    A2AError,
    A2ARequest,
    APIKeySecurityScheme,
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
    Artifact,
    CancelTaskRequest,
    CancelTaskResponse,
    CancelTaskSuccessResponse,
    ContentTypeNotSupportedError,
    DataPart,
    FileBase,
    FilePart,
    FileWithBytes,
    FileWithUri,
    GetTaskPushNotificationConfigRequest,
    GetTaskPushNotificationConfigResponse,
    GetTaskPushNotificationConfigSuccessResponse,
    GetTaskRequest,
    GetTaskResponse,
    GetTaskSuccessResponse,
    In,
    InternalError,
    InvalidParamsError,
    InvalidRequestError,
    JSONParseError,
    JSONRPCError,
    JSONRPCErrorResponse,
    JSONRPCMessage,
    JSONRPCRequest,
    JSONRPCResponse,
    Message,
    MessageSendParams,
    MethodNotFoundError,
    OAuth2SecurityScheme,
    Part,
    PartBase,
    PushNotificationAuthenticationInfo,
    PushNotificationConfig,
    PushNotificationNotSupportedError,
    Role,
    SecurityScheme,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    SendStreamingMessageRequest,
    SendStreamingMessageResponse,
    SendStreamingMessageSuccessResponse,
    SetTaskPushNotificationConfigRequest,
    SetTaskPushNotificationConfigResponse,
    SetTaskPushNotificationConfigSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskNotCancelableError,
    TaskNotFoundError,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskResubscriptionRequest,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)


# --- Helper Data ---

MINIMAL_AGENT_SECURITY_SCHEME: dict[str, Any] = {
    'type': 'apiKey',
    'in': 'header',
    'name': 'X-API-KEY',
}

MINIMAL_AGENT_SKILL: dict[str, Any] = {
    'id': 'skill-123',
    'name': 'Recipe Finder',
    'description': 'Finds recipes',
    'tags': ['cooking'],
}
FULL_AGENT_SKILL: dict[str, Any] = {
    'id': 'skill-123',
    'name': 'Recipe Finder',
    'description': 'Finds recipes',
    'tags': ['cooking', 'food'],
    'examples': ['Find me a pasta recipe'],
    'inputModes': ['text/plain'],
    'outputModes': ['application/json'],
}

MINIMAL_AGENT_CARD: dict[str, Any] = {
    'capabilities': {},  # AgentCapabilities is required but can be empty
    'defaultInputModes': ['text/plain'],
    'defaultOutputModes': ['application/json'],
    'description': 'Test Agent',
    'name': 'TestAgent',
    'skills': [MINIMAL_AGENT_SKILL],
    'url': 'http://example.com/agent',
    'version': '1.0',
}

TEXT_PART_DATA: dict[str, Any] = {'kind': 'text', 'text': 'Hello'}
FILE_URI_PART_DATA: dict[str, Any] = {
    'kind': 'file',
    'file': {'uri': 'file:///path/to/file.txt', 'mimeType': 'text/plain'},
}
FILE_BYTES_PART_DATA: dict[str, Any] = {
    'kind': 'file',
    'file': {'bytes': 'aGVsbG8=', 'name': 'hello.txt'},  # base64 for "hello"
}
DATA_PART_DATA: dict[str, Any] = {'kind': 'data', 'data': {'key': 'value'}}

MINIMAL_MESSAGE_USER: dict[str, Any] = {
    'role': 'user',
    'parts': [TEXT_PART_DATA],
    'messageId': 'msg-123',
    'kind': 'message',
}

AGENT_MESSAGE_WITH_FILE: dict[str, Any] = {
    'role': 'agent',
    'parts': [TEXT_PART_DATA, FILE_URI_PART_DATA],
    'metadata': {'timestamp': 'now'},
    'messageId': 'msg-456',
}

MINIMAL_TASK_STATUS: dict[str, Any] = {'state': 'submitted'}
FULL_TASK_STATUS: dict[str, Any] = {
    'state': 'working',
    'message': MINIMAL_MESSAGE_USER,
    'timestamp': '2023-10-27T10:00:00Z',
}

MINIMAL_TASK: dict[str, Any] = {
    'id': 'task-abc',
    'contextId': 'session-xyz',
    'status': MINIMAL_TASK_STATUS,
    'kind': 'task',
}
FULL_TASK: dict[str, Any] = {
    'id': 'task-abc',
    'contextId': 'session-xyz',
    'status': FULL_TASK_STATUS,
    'history': [MINIMAL_MESSAGE_USER, AGENT_MESSAGE_WITH_FILE],
    'artifacts': [
        {
            'artifactId': 'artifact-123',
            'parts': [DATA_PART_DATA],
            'name': 'result_data',
        }
    ],
    'metadata': {'priority': 'high'},
    'kind': 'task',
}

MINIMAL_TASK_ID_PARAMS: dict[str, Any] = {'id': 'task-123'}
FULL_TASK_ID_PARAMS: dict[str, Any] = {
    'id': 'task-456',
    'metadata': {'source': 'test'},
}

JSONRPC_ERROR_DATA: dict[str, Any] = {
    'code': -32600,
    'message': 'Invalid Request',
}
JSONRPC_SUCCESS_RESULT: dict[str, Any] = {'status': 'ok', 'data': [1, 2, 3]}

# --- Test Functions ---


def test_security_scheme_valid():
    scheme = SecurityScheme.model_validate(MINIMAL_AGENT_SECURITY_SCHEME)
    assert isinstance(scheme.root, APIKeySecurityScheme)
    assert scheme.root.type == 'apiKey'
    assert scheme.root.in_ == In.header
    assert scheme.root.name == 'X-API-KEY'


def test_security_scheme_invalid():
    with pytest.raises(ValidationError):
        APIKeySecurityScheme(
            name='my_api_key',
        )  # Missing "in"  # type: ignore

        OAuth2SecurityScheme(
            description='OAuth2 scheme missing flows',
        )  # Missing "flows"


def test_agent_capabilities():
    caps = AgentCapabilities(
        streaming=None, stateTransitionHistory=None, pushNotifications=None
    )  # All optional
    assert caps.pushNotifications is None
    assert caps.stateTransitionHistory is None
    assert caps.streaming is None

    caps_full = AgentCapabilities(
        pushNotifications=True, stateTransitionHistory=False, streaming=True
    )
    assert caps_full.pushNotifications is True
    assert caps_full.stateTransitionHistory is False
    assert caps_full.streaming is True


def test_agent_provider():
    provider = AgentProvider(organization='Test Org', url='http://test.org')
    assert provider.organization == 'Test Org'
    assert provider.url == 'http://test.org'

    with pytest.raises(ValidationError):
        AgentProvider(organization='Test Org')  # Missing url  # type: ignore


def test_agent_skill_valid():
    skill = AgentSkill(**MINIMAL_AGENT_SKILL)
    assert skill.id == 'skill-123'
    assert skill.name == 'Recipe Finder'
    assert skill.description == 'Finds recipes'
    assert skill.tags == ['cooking']
    assert skill.examples is None

    skill_full = AgentSkill(**FULL_AGENT_SKILL)
    assert skill_full.examples == ['Find me a pasta recipe']
    assert skill_full.inputModes == ['text/plain']


def test_agent_skill_invalid():
    with pytest.raises(ValidationError):
        AgentSkill(
            id='abc', name='n', description='d'
        )  # Missing tags  # type: ignore

    AgentSkill(
        **MINIMAL_AGENT_SKILL,
        invalid_extra='foo',  # type: ignore
    )  # Extra field


def test_agent_card_valid():
    card = AgentCard(**MINIMAL_AGENT_CARD)
    assert card.name == 'TestAgent'
    assert card.version == '1.0'
    assert len(card.skills) == 1
    assert card.skills[0].id == 'skill-123'
    assert card.provider is None  # Optional


def test_agent_card_invalid():
    bad_card_data = MINIMAL_AGENT_CARD.copy()
    del bad_card_data['name']
    with pytest.raises(ValidationError):
        AgentCard(**bad_card_data)  # Missing name


# --- Test Parts ---


def test_text_part():
    part = TextPart(**TEXT_PART_DATA)
    assert part.kind == 'text'
    assert part.text == 'Hello'
    assert part.metadata is None

    with pytest.raises(ValidationError):
        TextPart(type='text')  # Missing text # type: ignore
    with pytest.raises(ValidationError):
        TextPart(
            kind='file',  # type: ignore
            text='hello',
        )  # Wrong type literal


def test_file_part_variants():
    # URI variant
    file_uri = FileWithUri(
        uri='file:///path/to/file.txt', mimeType='text/plain'
    )
    part_uri = FilePart(kind='file', file=file_uri)
    assert isinstance(part_uri.file, FileWithUri)
    assert part_uri.file.uri == 'file:///path/to/file.txt'
    assert part_uri.file.mimeType == 'text/plain'
    assert not hasattr(part_uri.file, 'bytes')

    # Bytes variant
    file_bytes = FileWithBytes(bytes='aGVsbG8=', name='hello.txt')
    part_bytes = FilePart(kind='file', file=file_bytes)
    assert isinstance(part_bytes.file, FileWithBytes)
    assert part_bytes.file.bytes == 'aGVsbG8='
    assert part_bytes.file.name == 'hello.txt'
    assert not hasattr(part_bytes.file, 'uri')

    # Test deserialization directly
    part_uri_deserialized = FilePart.model_validate(FILE_URI_PART_DATA)
    assert isinstance(part_uri_deserialized.file, FileWithUri)
    assert part_uri_deserialized.file.uri == 'file:///path/to/file.txt'

    part_bytes_deserialized = FilePart.model_validate(FILE_BYTES_PART_DATA)
    assert isinstance(part_bytes_deserialized.file, FileWithBytes)
    assert part_bytes_deserialized.file.bytes == 'aGVsbG8='

    # Invalid - wrong type literal
    with pytest.raises(ValidationError):
        FilePart(kind='text', file=file_uri)  # type: ignore

    FilePart(**FILE_URI_PART_DATA, extra='extra')  # type: ignore


def test_data_part():
    part = DataPart(**DATA_PART_DATA)
    assert part.kind == 'data'
    assert part.data == {'key': 'value'}

    with pytest.raises(ValidationError):
        DataPart(type='data')  # Missing data  # type: ignore


def test_part_root_model():
    # Test deserialization of the Union RootModel
    part_text = Part.model_validate(TEXT_PART_DATA)
    assert isinstance(part_text.root, TextPart)
    assert part_text.root.text == 'Hello'

    part_file = Part.model_validate(FILE_URI_PART_DATA)
    assert isinstance(part_file.root, FilePart)
    assert isinstance(part_file.root.file, FileWithUri)

    part_data = Part.model_validate(DATA_PART_DATA)
    assert isinstance(part_data.root, DataPart)
    assert part_data.root.data == {'key': 'value'}

    # Test serialization
    assert part_text.model_dump(exclude_none=True) == TEXT_PART_DATA
    assert part_file.model_dump(exclude_none=True) == FILE_URI_PART_DATA
    assert part_data.model_dump(exclude_none=True) == DATA_PART_DATA


# --- Test Message and Task ---


def test_message():
    msg = Message(**MINIMAL_MESSAGE_USER)
    assert msg.role == Role.user
    assert len(msg.parts) == 1
    assert isinstance(
        msg.parts[0].root, TextPart
    )  # Access root for RootModel Part
    assert msg.metadata is None

    msg_agent = Message(**AGENT_MESSAGE_WITH_FILE)
    assert msg_agent.role == Role.agent
    assert len(msg_agent.parts) == 2
    assert isinstance(msg_agent.parts[1].root, FilePart)
    assert msg_agent.metadata == {'timestamp': 'now'}

    with pytest.raises(ValidationError):
        Message(
            role='invalid_role',  # type: ignore
            parts=[TEXT_PART_DATA],  # type: ignore
        )  # Invalid enum
    with pytest.raises(ValidationError):
        Message(role=Role.user)  # Missing parts  # type: ignore


def test_task_status():
    status = TaskStatus(**MINIMAL_TASK_STATUS)
    assert status.state == TaskState.submitted
    assert status.message is None
    assert status.timestamp is None

    status_full = TaskStatus(**FULL_TASK_STATUS)
    assert status_full.state == TaskState.working
    assert isinstance(status_full.message, Message)
    assert status_full.timestamp == '2023-10-27T10:00:00Z'

    with pytest.raises(ValidationError):
        TaskStatus(state='invalid_state')  # Invalid enum  # type: ignore


def test_task():
    task = Task(**MINIMAL_TASK)
    assert task.id == 'task-abc'
    assert task.contextId == 'session-xyz'
    assert task.status.state == TaskState.submitted
    assert task.history is None
    assert task.artifacts is None
    assert task.metadata is None

    task_full = Task(**FULL_TASK)
    assert task_full.id == 'task-abc'
    assert task_full.status.state == TaskState.working
    assert task_full.history is not None and len(task_full.history) == 2
    assert isinstance(task_full.history[0], Message)
    assert task_full.artifacts is not None and len(task_full.artifacts) == 1
    assert isinstance(task_full.artifacts[0], Artifact)
    assert task_full.artifacts[0].name == 'result_data'
    assert task_full.metadata == {'priority': 'high'}

    with pytest.raises(ValidationError):
        Task(id='abc', sessionId='xyz')  # Missing status # type: ignore


# --- Test JSON-RPC Structures ---


def test_jsonrpc_error():
    err = JSONRPCError(code=-32600, message='Invalid Request')
    assert err.code == -32600
    assert err.message == 'Invalid Request'
    assert err.data is None

    err_data = JSONRPCError(
        code=-32001, message='Task not found', data={'taskId': '123'}
    )
    assert err_data.code == -32001
    assert err_data.data == {'taskId': '123'}


def test_jsonrpc_request():
    req = JSONRPCRequest(jsonrpc='2.0', method='test_method', id=1)
    assert req.jsonrpc == '2.0'
    assert req.method == 'test_method'
    assert req.id == 1
    assert req.params is None

    req_params = JSONRPCRequest(
        jsonrpc='2.0', method='add', params={'a': 1, 'b': 2}, id='req-1'
    )
    assert req_params.params == {'a': 1, 'b': 2}
    assert req_params.id == 'req-1'

    with pytest.raises(ValidationError):
        JSONRPCRequest(
            jsonrpc='1.0',  # type: ignore
            method='m',
            id=1,
        )  # Wrong version
    with pytest.raises(ValidationError):
        JSONRPCRequest(jsonrpc='2.0', id=1)  # Missing method  # type: ignore


def test_jsonrpc_error_response():
    err_obj = JSONRPCError(**JSONRPC_ERROR_DATA)
    resp = JSONRPCErrorResponse(jsonrpc='2.0', error=err_obj, id='err-1')
    assert resp.jsonrpc == '2.0'
    assert resp.id == 'err-1'
    assert resp.error.code == -32600
    assert resp.error.message == 'Invalid Request'

    with pytest.raises(ValidationError):
        JSONRPCErrorResponse(
            jsonrpc='2.0', id='err-1'
        )  # Missing error # type: ignore


def test_jsonrpc_response_root_model() -> None:
    # Success case
    success_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'result': MINIMAL_TASK,
        'id': 1,
    }
    resp_success = JSONRPCResponse.model_validate(success_data)
    assert isinstance(resp_success.root, SendMessageSuccessResponse)
    assert resp_success.root.result == Task(**MINIMAL_TASK)

    # Error case
    error_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'error': JSONRPC_ERROR_DATA,
        'id': 'err-1',
    }
    resp_error = JSONRPCResponse.model_validate(error_data)
    assert isinstance(resp_error.root, JSONRPCErrorResponse)
    assert resp_error.root.error.code == -32600
    # Note: .model_dump() might serialize the nested error model
    assert resp_error.model_dump(exclude_none=True) == error_data

    # Invalid case (neither success nor error structure)
    with pytest.raises(ValidationError):
        JSONRPCResponse.model_validate({'jsonrpc': '2.0', 'id': 1})


# --- Test Request/Response Wrappers ---


def test_send_message_request() -> None:
    params = MessageSendParams(message=Message(**MINIMAL_MESSAGE_USER))
    req_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'method': 'message/send',
        'params': params.model_dump(),
        'id': 5,
    }
    req = SendMessageRequest.model_validate(req_data)
    assert req.method == 'message/send'
    assert isinstance(req.params, MessageSendParams)
    assert req.params.message.role == Role.user

    with pytest.raises(ValidationError):  # Wrong method literal
        SendMessageRequest.model_validate(
            {**req_data, 'method': 'wrong/method'}
        )


def test_send_subscribe_request() -> None:
    params = MessageSendParams(message=Message(**MINIMAL_MESSAGE_USER))
    req_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'method': 'message/stream',
        'params': params.model_dump(),
        'id': 5,
    }
    req = SendStreamingMessageRequest.model_validate(req_data)
    assert req.method == 'message/stream'
    assert isinstance(req.params, MessageSendParams)
    assert req.params.message.role == Role.user

    with pytest.raises(ValidationError):  # Wrong method literal
        SendStreamingMessageRequest.model_validate(
            {**req_data, 'method': 'wrong/method'}
        )


def test_get_task_request() -> None:
    params = TaskQueryParams(id='task-1', historyLength=2)
    req_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'method': 'tasks/get',
        'params': params.model_dump(),
        'id': 5,
    }
    req = GetTaskRequest.model_validate(req_data)
    assert req.method == 'tasks/get'
    assert isinstance(req.params, TaskQueryParams)
    assert req.params.id == 'task-1'
    assert req.params.historyLength == 2

    with pytest.raises(ValidationError):  # Wrong method literal
        GetTaskRequest.model_validate({**req_data, 'method': 'wrong/method'})


def test_cancel_task_request() -> None:
    params = TaskIdParams(id='task-1')
    req_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'method': 'tasks/cancel',
        'params': params.model_dump(),
        'id': 5,
    }
    req = CancelTaskRequest.model_validate(req_data)
    assert req.method == 'tasks/cancel'
    assert isinstance(req.params, TaskIdParams)
    assert req.params.id == 'task-1'

    with pytest.raises(ValidationError):  # Wrong method literal
        CancelTaskRequest.model_validate({**req_data, 'method': 'wrong/method'})


def test_get_task_response() -> None:
    resp_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'result': MINIMAL_TASK,
        'id': 'resp-1',
    }
    resp = GetTaskResponse.model_validate(resp_data)
    assert resp.root.id == 'resp-1'
    assert isinstance(resp.root, GetTaskSuccessResponse)
    assert isinstance(resp.root.result, Task)
    assert resp.root.result.id == 'task-abc'

    with pytest.raises(ValidationError):  # Result is not a Task
        GetTaskResponse.model_validate(
            {'jsonrpc': '2.0', 'result': {'wrong': 'data'}, 'id': 1}
        )

    resp_data_err: dict[str, Any] = {
        'jsonrpc': '2.0',
        'error': JSONRPCError(**TaskNotFoundError().model_dump()),
        'id': 'resp-1',
    }
    resp_err = GetTaskResponse.model_validate(resp_data_err)
    assert resp_err.root.id == 'resp-1'
    assert isinstance(resp_err.root, JSONRPCErrorResponse)
    assert resp_err.root.error is not None
    assert isinstance(resp_err.root.error, JSONRPCError)


def test_send_message_response() -> None:
    resp_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'result': MINIMAL_TASK,
        'id': 'resp-1',
    }
    resp = SendMessageResponse.model_validate(resp_data)
    assert resp.root.id == 'resp-1'
    assert isinstance(resp.root, SendMessageSuccessResponse)
    assert isinstance(resp.root.result, Task)
    assert resp.root.result.id == 'task-abc'

    with pytest.raises(ValidationError):  # Result is not a Task
        SendMessageResponse.model_validate(
            {'jsonrpc': '2.0', 'result': {'wrong': 'data'}, 'id': 1}
        )

    resp_data_err: dict[str, Any] = {
        'jsonrpc': '2.0',
        'error': JSONRPCError(**TaskNotFoundError().model_dump()),
        'id': 'resp-1',
    }
    resp_err = SendMessageResponse.model_validate(resp_data_err)
    assert resp_err.root.id == 'resp-1'
    assert isinstance(resp_err.root, JSONRPCErrorResponse)
    assert resp_err.root.error is not None
    assert isinstance(resp_err.root.error, JSONRPCError)


def test_cancel_task_response() -> None:
    resp_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'result': MINIMAL_TASK,
        'id': 1,
    }
    resp = CancelTaskResponse.model_validate(resp_data)
    assert resp.root.id == 1
    assert isinstance(resp.root, CancelTaskSuccessResponse)
    assert isinstance(resp.root.result, Task)
    assert resp.root.result.id == 'task-abc'

    resp_data_err: dict[str, Any] = {
        'jsonrpc': '2.0',
        'error': JSONRPCError(**TaskNotFoundError().model_dump()),
        'id': 'resp-1',
    }
    resp_err = CancelTaskResponse.model_validate(resp_data_err)
    assert resp_err.root.id == 'resp-1'
    assert isinstance(resp_err.root, JSONRPCErrorResponse)
    assert resp_err.root.error is not None
    assert isinstance(resp_err.root.error, JSONRPCError)


def test_send_message_streaming_status_update_response() -> None:
    task_status_update_event_data: dict[str, Any] = {
        'status': MINIMAL_TASK_STATUS,
        'taskId': '1',
        'contextId': '2',
        'final': False,
        'kind': 'status-update',
    }

    event_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'id': 1,
        'result': task_status_update_event_data,
    }
    response = SendStreamingMessageResponse.model_validate(event_data)
    assert response.root.id == 1
    assert isinstance(response.root, SendStreamingMessageSuccessResponse)
    assert isinstance(response.root.result, TaskStatusUpdateEvent)
    assert response.root.result.status.state == TaskState.submitted
    assert response.root.result.taskId == '1'
    assert not response.root.result.final

    with pytest.raises(
        ValidationError
    ):  # Result is not a TaskStatusUpdateEvent
        SendStreamingMessageResponse.model_validate(
            {'jsonrpc': '2.0', 'result': {'wrong': 'data'}, 'id': 1}
        )

    event_data = {
        'jsonrpc': '2.0',
        'id': 1,
        'result': {**task_status_update_event_data, 'final': True},
    }
    response = SendStreamingMessageResponse.model_validate(event_data)
    assert response.root.id == 1
    assert isinstance(response.root, SendStreamingMessageSuccessResponse)
    assert isinstance(response.root.result, TaskStatusUpdateEvent)
    assert response.root.result.final

    resp_data_err: dict[str, Any] = {
        'jsonrpc': '2.0',
        'error': JSONRPCError(**TaskNotFoundError().model_dump()),
        'id': 'resp-1',
    }
    resp_err = SendStreamingMessageResponse.model_validate(resp_data_err)
    assert resp_err.root.id == 'resp-1'
    assert isinstance(resp_err.root, JSONRPCErrorResponse)
    assert resp_err.root.error is not None
    assert isinstance(resp_err.root.error, JSONRPCError)


def test_send_message_streaming_artifact_update_response() -> None:
    text_part = TextPart(**TEXT_PART_DATA)
    data_part = DataPart(**DATA_PART_DATA)
    artifact = Artifact(
        artifactId='artifact-123',
        name='result_data',
        parts=[Part(root=text_part), Part(root=data_part)],
    )
    task_artifact_update_event_data: dict[str, Any] = {
        'artifact': artifact,
        'taskId': 'task_id',
        'contextId': '2',
        'append': False,
        'lastChunk': True,
        'kind': 'artifact-update',
    }
    event_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'id': 1,
        'result': task_artifact_update_event_data,
    }
    response = SendStreamingMessageResponse.model_validate(event_data)
    assert response.root.id == 1
    assert isinstance(response.root, SendStreamingMessageSuccessResponse)
    assert isinstance(response.root.result, TaskArtifactUpdateEvent)
    assert response.root.result.artifact.artifactId == 'artifact-123'
    assert response.root.result.artifact.name == 'result_data'
    assert response.root.result.taskId == 'task_id'
    assert not response.root.result.append
    assert response.root.result.lastChunk
    assert len(response.root.result.artifact.parts) == 2
    assert isinstance(response.root.result.artifact.parts[0].root, TextPart)
    assert isinstance(response.root.result.artifact.parts[1].root, DataPart)


def test_set_task_push_notification_response() -> None:
    task_push_config = TaskPushNotificationConfig(
        taskId='t2',
        pushNotificationConfig=PushNotificationConfig(
            url='https://example.com', token='token'
        ),
    )
    resp_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'result': task_push_config.model_dump(),
        'id': 1,
    }
    resp = SetTaskPushNotificationConfigResponse.model_validate(resp_data)
    assert resp.root.id == 1
    assert isinstance(resp.root, SetTaskPushNotificationConfigSuccessResponse)
    assert isinstance(resp.root.result, TaskPushNotificationConfig)
    assert resp.root.result.taskId == 't2'
    assert resp.root.result.pushNotificationConfig.url == 'https://example.com'
    assert resp.root.result.pushNotificationConfig.token == 'token'
    assert resp.root.result.pushNotificationConfig.authentication is None

    auth_info_dict: dict[str, Any] = {
        'schemes': ['Bearer', 'Basic'],
        'credentials': 'user:pass',
    }
    task_push_config.pushNotificationConfig.authentication = (
        PushNotificationAuthenticationInfo(**auth_info_dict)
    )
    resp_data = {
        'jsonrpc': '2.0',
        'result': task_push_config.model_dump(),
        'id': 1,
    }
    resp = SetTaskPushNotificationConfigResponse.model_validate(resp_data)
    assert isinstance(resp.root, SetTaskPushNotificationConfigSuccessResponse)
    assert resp.root.result.pushNotificationConfig.authentication is not None
    assert resp.root.result.pushNotificationConfig.authentication.schemes == [
        'Bearer',
        'Basic',
    ]
    assert (
        resp.root.result.pushNotificationConfig.authentication.credentials
        == 'user:pass'
    )

    resp_data_err: dict[str, Any] = {
        'jsonrpc': '2.0',
        'error': JSONRPCError(**TaskNotFoundError().model_dump()),
        'id': 'resp-1',
    }
    resp_err = SetTaskPushNotificationConfigResponse.model_validate(
        resp_data_err
    )
    assert resp_err.root.id == 'resp-1'
    assert isinstance(resp_err.root, JSONRPCErrorResponse)
    assert resp_err.root.error is not None
    assert isinstance(resp_err.root.error, JSONRPCError)


def test_get_task_push_notification_response() -> None:
    task_push_config = TaskPushNotificationConfig(
        taskId='t2',
        pushNotificationConfig=PushNotificationConfig(
            url='https://example.com', token='token'
        ),
    )
    resp_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'result': task_push_config.model_dump(),
        'id': 1,
    }
    resp = GetTaskPushNotificationConfigResponse.model_validate(resp_data)
    assert resp.root.id == 1
    assert isinstance(resp.root, GetTaskPushNotificationConfigSuccessResponse)
    assert isinstance(resp.root.result, TaskPushNotificationConfig)
    assert resp.root.result.taskId == 't2'
    assert resp.root.result.pushNotificationConfig.url == 'https://example.com'
    assert resp.root.result.pushNotificationConfig.token == 'token'
    assert resp.root.result.pushNotificationConfig.authentication is None

    auth_info_dict: dict[str, Any] = {
        'schemes': ['Bearer', 'Basic'],
        'credentials': 'user:pass',
    }
    task_push_config.pushNotificationConfig.authentication = (
        PushNotificationAuthenticationInfo(**auth_info_dict)
    )
    resp_data = {
        'jsonrpc': '2.0',
        'result': task_push_config.model_dump(),
        'id': 1,
    }
    resp = GetTaskPushNotificationConfigResponse.model_validate(resp_data)
    assert isinstance(resp.root, GetTaskPushNotificationConfigSuccessResponse)
    assert resp.root.result.pushNotificationConfig.authentication is not None
    assert resp.root.result.pushNotificationConfig.authentication.schemes == [
        'Bearer',
        'Basic',
    ]
    assert (
        resp.root.result.pushNotificationConfig.authentication.credentials
        == 'user:pass'
    )

    resp_data_err: dict[str, Any] = {
        'jsonrpc': '2.0',
        'error': JSONRPCError(**TaskNotFoundError().model_dump()),
        'id': 'resp-1',
    }
    resp_err = GetTaskPushNotificationConfigResponse.model_validate(
        resp_data_err
    )
    assert resp_err.root.id == 'resp-1'
    assert isinstance(resp_err.root, JSONRPCErrorResponse)
    assert resp_err.root.error is not None
    assert isinstance(resp_err.root.error, JSONRPCError)


# --- Test A2ARequest Root Model ---


def test_a2a_request_root_model() -> None:
    # SendMessageRequest case
    send_params = MessageSendParams(message=Message(**MINIMAL_MESSAGE_USER))
    send_req_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'method': 'message/send',
        'params': send_params.model_dump(),
        'id': 1,
    }
    a2a_req_send = A2ARequest.model_validate(send_req_data)
    assert isinstance(a2a_req_send.root, SendMessageRequest)
    assert a2a_req_send.root.method == 'message/send'

    # SendStreamingMessageRequest case
    send_subs_req_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'method': 'message/stream',
        'params': send_params.model_dump(),
        'id': 1,
    }
    a2a_req_send_subs = A2ARequest.model_validate(send_subs_req_data)
    assert isinstance(a2a_req_send_subs.root, SendStreamingMessageRequest)
    assert a2a_req_send_subs.root.method == 'message/stream'

    # GetTaskRequest case
    get_params = TaskQueryParams(id='t2')
    get_req_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'method': 'tasks/get',
        'params': get_params.model_dump(),
        'id': 2,
    }
    a2a_req_get = A2ARequest.model_validate(get_req_data)
    assert isinstance(a2a_req_get.root, GetTaskRequest)
    assert a2a_req_get.root.method == 'tasks/get'

    # CancelTaskRequest case
    id_params = TaskIdParams(id='t2')
    cancel_req_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'method': 'tasks/cancel',
        'params': id_params.model_dump(),
        'id': 2,
    }
    a2a_req_cancel = A2ARequest.model_validate(cancel_req_data)
    assert isinstance(a2a_req_cancel.root, CancelTaskRequest)
    assert a2a_req_cancel.root.method == 'tasks/cancel'

    # SetTaskPushNotificationConfigRequest
    task_push_config = TaskPushNotificationConfig(
        taskId='t2',
        pushNotificationConfig=PushNotificationConfig(
            url='https://example.com', token='token'
        ),
    )
    set_push_notif_req_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'method': 'tasks/pushNotificationConfig/set',
        'params': task_push_config.model_dump(),
        'taskId': 2,
    }
    a2a_req_set_push_req = A2ARequest.model_validate(set_push_notif_req_data)
    assert isinstance(
        a2a_req_set_push_req.root, SetTaskPushNotificationConfigRequest
    )
    assert isinstance(
        a2a_req_set_push_req.root.params, TaskPushNotificationConfig
    )
    assert (
        a2a_req_set_push_req.root.method == 'tasks/pushNotificationConfig/set'
    )

    # GetTaskPushNotificationConfigRequest
    id_params = TaskIdParams(id='t2')
    get_push_notif_req_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'method': 'tasks/pushNotificationConfig/get',
        'params': id_params.model_dump(),
        'taskId': 2,
    }
    a2a_req_get_push_req = A2ARequest.model_validate(get_push_notif_req_data)
    assert isinstance(
        a2a_req_get_push_req.root, GetTaskPushNotificationConfigRequest
    )
    assert isinstance(a2a_req_get_push_req.root.params, TaskIdParams)
    assert (
        a2a_req_get_push_req.root.method == 'tasks/pushNotificationConfig/get'
    )

    # TaskResubscriptionRequest
    task_resubscribe_req_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'method': 'tasks/resubscribe',
        'params': id_params.model_dump(),
        'id': 2,
    }
    a2a_req_task_resubscribe_req = A2ARequest.model_validate(
        task_resubscribe_req_data
    )
    assert isinstance(
        a2a_req_task_resubscribe_req.root, TaskResubscriptionRequest
    )
    assert isinstance(a2a_req_task_resubscribe_req.root.params, TaskIdParams)
    assert a2a_req_task_resubscribe_req.root.method == 'tasks/resubscribe'

    # Invalid method case
    invalid_req_data: dict[str, Any] = {
        'jsonrpc': '2.0',
        'method': 'invalid/method',
        'params': {},
        'id': 3,
    }
    with pytest.raises(ValidationError):
        A2ARequest.model_validate(invalid_req_data)


def test_content_type_not_supported_error():
    # Test ContentTypeNotSupportedError
    err = ContentTypeNotSupportedError(
        code=-32005, message='Incompatible content types'
    )
    assert err.code == -32005
    assert err.message == 'Incompatible content types'
    assert err.data is None

    with pytest.raises(ValidationError):  # Wrong code
        ContentTypeNotSupportedError(
            code=-32000,  # type: ignore
            message='Incompatible content types',
        )

    ContentTypeNotSupportedError(
        code=-32005,
        message='Incompatible content types',
        extra='extra',  # type: ignore
    )


def test_task_not_found_error():
    # Test TaskNotFoundError
    err2 = TaskNotFoundError(
        code=-32001, message='Task not found', data={'taskId': 'abc'}
    )
    assert err2.code == -32001
    assert err2.message == 'Task not found'
    assert err2.data == {'taskId': 'abc'}

    with pytest.raises(ValidationError):  # Wrong code
        TaskNotFoundError(code=-32000, message='Task not found')  # type: ignore

    TaskNotFoundError(code=-32001, message='Task not found', extra='extra')  # type: ignore


def test_push_notification_not_supported_error():
    # Test PushNotificationNotSupportedError
    err3 = PushNotificationNotSupportedError(data={'taskId': 'abc'})
    assert err3.code == -32003
    assert err3.message == 'Push Notification is not supported'
    assert err3.data == {'taskId': 'abc'}

    with pytest.raises(ValidationError):  # Wrong code
        PushNotificationNotSupportedError(
            code=-32000,  # type: ignore
            message='Push Notification is not available',
        )
    with pytest.raises(ValidationError):  # Extra field
        PushNotificationNotSupportedError(
            code=-32001,
            message='Push Notification is not available',
            extra='extra',  # type: ignore
        )


def test_internal_error():
    # Test InternalError
    err_internal = InternalError()
    assert err_internal.code == -32603
    assert err_internal.message == 'Internal error'
    assert err_internal.data is None

    err_internal_data = InternalError(
        code=-32603, message='Internal error', data={'details': 'stack trace'}
    )
    assert err_internal_data.data == {'details': 'stack trace'}

    with pytest.raises(ValidationError):  # Wrong code
        InternalError(code=-32000, message='Internal error')  # type: ignore

    InternalError(code=-32603, message='Internal error', extra='extra')  # type: ignore


def test_invalid_params_error():
    # Test InvalidParamsError
    err_params = InvalidParamsError()
    assert err_params.code == -32602
    assert err_params.message == 'Invalid parameters'
    assert err_params.data is None

    err_params_data = InvalidParamsError(
        code=-32602, message='Invalid parameters', data=['param1', 'param2']
    )
    assert err_params_data.data == ['param1', 'param2']

    with pytest.raises(ValidationError):  # Wrong code
        InvalidParamsError(code=-32000, message='Invalid parameters')  # type: ignore

    InvalidParamsError(
        code=-32602,
        message='Invalid parameters',
        extra='extra',  # type: ignore
    )


def test_invalid_request_error():
    # Test InvalidRequestError
    err_request = InvalidRequestError()
    assert err_request.code == -32600
    assert err_request.message == 'Request payload validation error'
    assert err_request.data is None

    err_request_data = InvalidRequestError(data={'field': 'missing'})
    assert err_request_data.data == {'field': 'missing'}

    with pytest.raises(ValidationError):  # Wrong code
        InvalidRequestError(
            code=-32000,  # type: ignore
            message='Request payload validation error',
        )

    InvalidRequestError(
        code=-32600,
        message='Request payload validation error',
        extra='extra',  # type: ignore
    )  # type: ignore


def test_json_parse_error():
    # Test JSONParseError
    err_parse = JSONParseError(code=-32700, message='Invalid JSON payload')
    assert err_parse.code == -32700
    assert err_parse.message == 'Invalid JSON payload'
    assert err_parse.data is None

    err_parse_data = JSONParseError(data={'foo': 'bar'})  # Explicit None data
    assert err_parse_data.data == {'foo': 'bar'}

    with pytest.raises(ValidationError):  # Wrong code
        JSONParseError(code=-32000, message='Invalid JSON payload')  # type: ignore

    JSONParseError(code=-32700, message='Invalid JSON payload', extra='extra')  # type: ignore


def test_method_not_found_error():
    # Test MethodNotFoundError
    err_parse = MethodNotFoundError()
    assert err_parse.code == -32601
    assert err_parse.message == 'Method not found'
    assert err_parse.data is None

    err_parse_data = JSONParseError(data={'foo': 'bar'})
    assert err_parse_data.data == {'foo': 'bar'}

    with pytest.raises(ValidationError):  # Wrong code
        JSONParseError(code=-32000, message='Invalid JSON payload')  # type: ignore

    JSONParseError(code=-32700, message='Invalid JSON payload', extra='extra')  # type: ignore


def test_task_not_cancelable_error():
    # Test TaskNotCancelableError
    err_parse = TaskNotCancelableError()
    assert err_parse.code == -32002
    assert err_parse.message == 'Task cannot be canceled'
    assert err_parse.data is None

    err_parse_data = JSONParseError(
        data={'foo': 'bar'}, message='not cancelled'
    )
    assert err_parse_data.data == {'foo': 'bar'}
    assert err_parse_data.message == 'not cancelled'

    with pytest.raises(ValidationError):  # Wrong code
        JSONParseError(code=-32000, message='Task cannot be canceled')  # type: ignore

    JSONParseError(
        code=-32700,
        message='Task cannot be canceled',
        extra='extra',  # type: ignore
    )


def test_unsupported_operation_error():
    # Test UnsupportedOperationError
    err_parse = UnsupportedOperationError()
    assert err_parse.code == -32004
    assert err_parse.message == 'This operation is not supported'
    assert err_parse.data is None

    err_parse_data = JSONParseError(
        data={'foo': 'bar'}, message='not supported'
    )
    assert err_parse_data.data == {'foo': 'bar'}
    assert err_parse_data.message == 'not supported'

    with pytest.raises(ValidationError):  # Wrong code
        JSONParseError(code=-32000, message='Unsupported')  # type: ignore

    JSONParseError(code=-32700, message='Unsupported', extra='extra')  # type: ignore


# --- Test TaskIdParams ---


def test_task_id_params_valid():
    """Tests successful validation of TaskIdParams."""
    # Minimal valid data
    params_min = TaskIdParams(**MINIMAL_TASK_ID_PARAMS)
    assert params_min.id == 'task-123'
    assert params_min.metadata is None

    # Full valid data
    params_full = TaskIdParams(**FULL_TASK_ID_PARAMS)
    assert params_full.id == 'task-456'
    assert params_full.metadata == {'source': 'test'}


def test_task_id_params_invalid():
    """Tests validation errors for TaskIdParams."""
    # Missing required 'id' field
    with pytest.raises(ValidationError) as excinfo_missing:
        TaskIdParams()  # type: ignore
    assert 'id' in str(
        excinfo_missing.value
    )  # Check that 'id' is mentioned in the error

    invalid_data = MINIMAL_TASK_ID_PARAMS.copy()
    invalid_data['extra_field'] = 'allowed'
    TaskIdParams(**invalid_data)  # type: ignore

    # Incorrect type for metadata (should be dict)
    invalid_metadata_type = {'id': 'task-789', 'metadata': 'not_a_dict'}
    with pytest.raises(ValidationError) as excinfo_type:
        TaskIdParams(**invalid_metadata_type)  # type: ignore
    assert 'metadata' in str(
        excinfo_type.value
    )  # Check that 'metadata' is mentioned


def test_task_push_notification_config() -> None:
    """Tests successful validation of TaskPushNotificationConfig."""
    auth_info_dict: dict[str, Any] = {
        'schemes': ['Bearer', 'Basic'],
        'credentials': 'user:pass',
    }
    auth_info = PushNotificationAuthenticationInfo(**auth_info_dict)

    push_notification_config = PushNotificationConfig(
        url='https://example.com', token='token', authentication=auth_info
    )
    assert push_notification_config.url == 'https://example.com'
    assert push_notification_config.token == 'token'
    assert push_notification_config.authentication == auth_info

    task_push_notification_config = TaskPushNotificationConfig(
        taskId='task-123', pushNotificationConfig=push_notification_config
    )
    assert task_push_notification_config.taskId == 'task-123'
    assert (
        task_push_notification_config.pushNotificationConfig
        == push_notification_config
    )
    assert task_push_notification_config.model_dump(exclude_none=True) == {
        'taskId': 'task-123',
        'pushNotificationConfig': {
            'url': 'https://example.com',
            'token': 'token',
            'authentication': {
                'schemes': ['Bearer', 'Basic'],
                'credentials': 'user:pass',
            },
        },
    }


def test_jsonrpc_message_valid():
    """Tests successful validation of JSONRPCMessage."""
    # With string ID
    msg_str_id = JSONRPCMessage(jsonrpc='2.0', id='req-1')
    assert msg_str_id.jsonrpc == '2.0'
    assert msg_str_id.id == 'req-1'

    # With integer ID (will be coerced to float by Pydantic for JSON number compatibility)
    msg_int_id = JSONRPCMessage(jsonrpc='2.0', id=1)
    assert msg_int_id.jsonrpc == '2.0'
    assert (
        msg_int_id.id == 1
    )  # Pydantic v2 keeps int if possible, but float is in type hint

    rpc_message = JSONRPCMessage(id=1)
    assert rpc_message.jsonrpc == '2.0'
    assert rpc_message.id == 1


def test_jsonrpc_message_invalid():
    """Tests validation errors for JSONRPCMessage."""
    # Incorrect jsonrpc version
    with pytest.raises(ValidationError):
        JSONRPCMessage(jsonrpc='1.0', id=1)  # type: ignore

    JSONRPCMessage(jsonrpc='2.0', id=1, extra_field='extra')  # type: ignore

    # Invalid ID type (e.g., list) - Pydantic should catch this based on type hints
    with pytest.raises(ValidationError):
        JSONRPCMessage(jsonrpc='2.0', id=[1, 2])  # type: ignore


def test_file_base_valid():
    """Tests successful validation of FileBase."""
    # No optional fields
    base1 = FileBase()
    assert base1.mimeType is None
    assert base1.name is None

    # With mimeType only
    base2 = FileBase(mimeType='image/png')
    assert base2.mimeType == 'image/png'
    assert base2.name is None

    # With name only
    base3 = FileBase(name='document.pdf')
    assert base3.mimeType is None
    assert base3.name == 'document.pdf'

    # With both fields
    base4 = FileBase(mimeType='application/json', name='data.json')
    assert base4.mimeType == 'application/json'
    assert base4.name == 'data.json'


def test_file_base_invalid():
    """Tests validation errors for FileBase."""
    FileBase(extra_field='allowed')  # type: ignore

    # Incorrect type for mimeType
    with pytest.raises(ValidationError) as excinfo_type_mime:
        FileBase(mimeType=123)  # type: ignore
    assert 'mimeType' in str(excinfo_type_mime.value)

    # Incorrect type for name
    with pytest.raises(ValidationError) as excinfo_type_name:
        FileBase(name=['list', 'is', 'wrong'])  # type: ignore
    assert 'name' in str(excinfo_type_name.value)


def test_part_base_valid() -> None:
    """Tests successful validation of PartBase."""
    # No optional fields (metadata is None)
    base1 = PartBase()
    assert base1.metadata is None

    # With metadata
    meta_data: dict[str, Any] = {'source': 'test', 'timestamp': 12345}
    base2 = PartBase(metadata=meta_data)
    assert base2.metadata == meta_data


def test_part_base_invalid():
    """Tests validation errors for PartBase."""
    PartBase(extra_field='allowed')  # type: ignore

    # Incorrect type for metadata (should be dict)
    with pytest.raises(ValidationError) as excinfo_type:
        PartBase(metadata='not_a_dict')  # type: ignore
    assert 'metadata' in str(excinfo_type.value)


def test_a2a_error_validation_and_serialization() -> None:
    """Tests validation and serialization of the A2AError RootModel."""

    # 1. Test JSONParseError
    json_parse_instance = JSONParseError()
    json_parse_data = json_parse_instance.model_dump(exclude_none=True)
    a2a_err_parse = A2AError.model_validate(json_parse_data)
    assert isinstance(a2a_err_parse.root, JSONParseError)

    # 2. Test InvalidRequestError
    invalid_req_instance = InvalidRequestError()
    invalid_req_data = invalid_req_instance.model_dump(exclude_none=True)
    a2a_err_invalid_req = A2AError.model_validate(invalid_req_data)
    assert isinstance(a2a_err_invalid_req.root, InvalidRequestError)

    # 3. Test MethodNotFoundError
    method_not_found_instance = MethodNotFoundError()
    method_not_found_data = method_not_found_instance.model_dump(
        exclude_none=True
    )
    a2a_err_method = A2AError.model_validate(method_not_found_data)
    assert isinstance(a2a_err_method.root, MethodNotFoundError)

    # 4. Test InvalidParamsError
    invalid_params_instance = InvalidParamsError()
    invalid_params_data = invalid_params_instance.model_dump(exclude_none=True)
    a2a_err_params = A2AError.model_validate(invalid_params_data)
    assert isinstance(a2a_err_params.root, InvalidParamsError)

    # 5. Test InternalError
    internal_err_instance = InternalError()
    internal_err_data = internal_err_instance.model_dump(exclude_none=True)
    a2a_err_internal = A2AError.model_validate(internal_err_data)
    assert isinstance(a2a_err_internal.root, InternalError)

    # 6. Test TaskNotFoundError
    task_not_found_instance = TaskNotFoundError(data={'taskId': 't1'})
    task_not_found_data = task_not_found_instance.model_dump(exclude_none=True)
    a2a_err_task_nf = A2AError.model_validate(task_not_found_data)
    assert isinstance(a2a_err_task_nf.root, TaskNotFoundError)

    # 7. Test TaskNotCancelableError
    task_not_cancelable_instance = TaskNotCancelableError()
    task_not_cancelable_data = task_not_cancelable_instance.model_dump(
        exclude_none=True
    )
    a2a_err_task_nc = A2AError.model_validate(task_not_cancelable_data)
    assert isinstance(a2a_err_task_nc.root, TaskNotCancelableError)

    # 8. Test PushNotificationNotSupportedError
    push_not_supported_instance = PushNotificationNotSupportedError()
    push_not_supported_data = push_not_supported_instance.model_dump(
        exclude_none=True
    )
    a2a_err_push_ns = A2AError.model_validate(push_not_supported_data)
    assert isinstance(a2a_err_push_ns.root, PushNotificationNotSupportedError)

    # 9. Test UnsupportedOperationError
    unsupported_op_instance = UnsupportedOperationError()
    unsupported_op_data = unsupported_op_instance.model_dump(exclude_none=True)
    a2a_err_unsupported = A2AError.model_validate(unsupported_op_data)
    assert isinstance(a2a_err_unsupported.root, UnsupportedOperationError)

    # 10. Test ContentTypeNotSupportedError
    content_type_err_instance = ContentTypeNotSupportedError()
    content_type_err_data = content_type_err_instance.model_dump(
        exclude_none=True
    )
    a2a_err_content = A2AError.model_validate(content_type_err_data)
    assert isinstance(a2a_err_content.root, ContentTypeNotSupportedError)

    # 11. Test invalid data (doesn't match any known error code/structure)
    invalid_data: dict[str, Any] = {'code': -99999, 'message': 'Unknown error'}
    with pytest.raises(ValidationError):
        A2AError.model_validate(invalid_data)
````

## File: SECURITY.md
````markdown
# Security Policy

To report a security issue, please use [g.co/vulnz](https://g.co/vulnz).

The Google Security Team will respond within 5 working days of your report on g.co/vulnz.

We use g.co/vulnz for our intake, and do coordination and disclosure here using GitHub Security Advisory to privately discuss and fix the issue.
````

## File: src/a2a/server/apps/__init__.py
````python
"""HTTP application components for the A2A server."""

from a2a.server.apps.starlette_app import A2AStarletteApplication


__all__ = ['A2AStarletteApplication']
````

## File: src/a2a/server/events/__init__.py
````python
"""Event handling components for the A2A server."""

from a2a.server.events.event_consumer import EventConsumer
from a2a.server.events.event_queue import Event, EventQueue
from a2a.server.events.in_memory_queue_manager import InMemoryQueueManager
from a2a.server.events.queue_manager import (
    NoTaskQueue,
    QueueManager,
    TaskQueueExists,
)


__all__ = [
    'Event',
    'EventConsumer',
    'EventQueue',
    'InMemoryQueueManager',
    'NoTaskQueue',
    'QueueManager',
    'TaskQueueExists',
]
````

## File: src/a2a/server/events/queue_manager.py
````python
from abc import ABC, abstractmethod

from a2a.server.events.event_queue import EventQueue


class QueueManager(ABC):
    """Interface for managing the event queue lifecycles per task."""

    @abstractmethod
    async def add(self, task_id: str, queue: EventQueue):
        """Adds a new event queue associated with a task ID."""

    @abstractmethod
    async def get(self, task_id: str) -> EventQueue | None:
        """Retrieves the event queue for a task ID."""

    @abstractmethod
    async def tap(self, task_id: str) -> EventQueue | None:
        """Creates a child event queue (tap) for an existing task ID."""

    @abstractmethod
    async def close(self, task_id: str):
        """Closes and removes the event queue for a task ID."""

    @abstractmethod
    async def create_or_tap(self, task_id: str) -> EventQueue:
        """Creates a queue if one doesn't exist, otherwise taps the existing one."""


class TaskQueueExists(Exception):
    """Exception raised when attempting to add a queue for a task ID that already exists."""


class NoTaskQueue(Exception):
    """Exception raised when attempting to access or close a queue for a task ID that does not exist."""
````

## File: src/a2a/server/tasks/inmemory_task_store.py
````python
import asyncio
import logging

from a2a.server.tasks.task_store import TaskStore
from a2a.types import Task


logger = logging.getLogger(__name__)


class InMemoryTaskStore(TaskStore):
    """In-memory implementation of TaskStore.

    Stores task objects in a dictionary in memory. Task data is lost when the
    server process stops.
    """

    def __init__(self) -> None:
        """Initializes the InMemoryTaskStore."""
        logger.debug('Initializing InMemoryTaskStore')
        self.tasks: dict[str, Task] = {}
        self.lock = asyncio.Lock()

    async def save(self, task: Task) -> None:
        """Saves or updates a task in the in-memory store."""
        async with self.lock:
            self.tasks[task.id] = task
            logger.debug('Task %s saved successfully.', task.id)

    async def get(self, task_id: str) -> Task | None:
        """Retrieves a task from the in-memory store by ID."""
        async with self.lock:
            logger.debug('Attempting to get task with id: %s', task_id)
            task = self.tasks.get(task_id)
            if task:
                logger.debug('Task %s retrieved successfully.', task_id)
            else:
                logger.debug('Task %s not found in store.', task_id)
            return task

    async def delete(self, task_id: str) -> None:
        """Deletes a task from the in-memory store by ID."""
        async with self.lock:
            logger.debug('Attempting to delete task with id: %s', task_id)
            if task_id in self.tasks:
                del self.tasks[task_id]
                logger.debug('Task %s deleted successfully.', task_id)
            else:
                logger.warning(
                    'Attempted to delete nonexistent task with id: %s', task_id
                )
````

## File: src/a2a/utils/errors.py
````python
"""Custom exceptions for A2A server-side errors."""

from a2a.types import (
    ContentTypeNotSupportedError,
    InternalError,
    InvalidAgentResponseError,
    InvalidParamsError,
    InvalidRequestError,
    JSONParseError,
    JSONRPCError,
    MethodNotFoundError,
    PushNotificationNotSupportedError,
    TaskNotCancelableError,
    TaskNotFoundError,
    UnsupportedOperationError,
)


class A2AServerError(Exception):
    """Base exception for A2A Server errors."""


class MethodNotImplementedError(A2AServerError):
    """Exception raised for methods that are not implemented by the server handler."""

    def __init__(
        self, message: str = 'This method is not implemented by the server'
    ):
        """Initializes the MethodNotImplementedError.

        Args:
            message: A descriptive error message.
        """
        self.message = message
        super().__init__(f'Not Implemented operation Error: {message}')


class ServerError(Exception):
    """Wrapper exception for A2A or JSON-RPC errors originating from the server's logic.

    This exception is used internally by request handlers and other server components
    to signal a specific error that should be formatted as a JSON-RPC error response.
    """

    def __init__(
        self,
        error: (
            JSONRPCError
            | JSONParseError
            | InvalidRequestError
            | MethodNotFoundError
            | InvalidParamsError
            | InternalError
            | TaskNotFoundError
            | TaskNotCancelableError
            | PushNotificationNotSupportedError
            | UnsupportedOperationError
            | ContentTypeNotSupportedError
            | InvalidAgentResponseError
            | None
        ),
    ):
        """Initializes the ServerError.

        Args:
            error: The specific A2A or JSON-RPC error model instance.
                   If None, an `InternalError` will be used when formatting the response.
        """
        self.error = error
````

## File: src/a2a/utils/telemetry.py
````python
"""OpenTelemetry Tracing Utilities for A2A Python SDK.

This module provides decorators to simplify the integration of OpenTelemetry
tracing into Python applications. It offers `trace_function` for instrumenting
individual functions (both synchronous and asynchronous) and `trace_class`
for instrumenting multiple methods within a class.

The tracer is initialized with the module name and version defined by
`INSTRUMENTING_MODULE_NAME` ('a2a-python-sdk') and
`INSTRUMENTING_MODULE_VERSION` ('1.0.0').

Features:
- Automatic span creation for decorated functions/methods.
- Support for both synchronous and asynchronous functions.
- Default span naming based on module and function/class/method name.
- Customizable span names, kinds, and static attributes.
- Dynamic attribute setting via an `attribute_extractor` callback.
- Automatic recording of exceptions and setting of span status.
- Selective method tracing in classes using include/exclude lists.

Usage:
    For a single function:
    ```python
    from your_module import trace_function


    @trace_function
    def my_function():
        # ...
        pass


    @trace_function(span_name='custom.op', kind=SpanKind.CLIENT)
    async def my_async_function():
        # ...
        pass
    ```

    For a class:
    ```python
    from your_module import trace_class


    @trace_class(exclude_list=['internal_method'])
    class MyService:
        def public_api(self, user_id):
            # This method will be traced
            pass

        def internal_method(self):
            # This method will not be traced
            pass
    ```
"""

import functools
import inspect
import logging

from opentelemetry import trace
from opentelemetry.trace import SpanKind as _SpanKind
from opentelemetry.trace import StatusCode


SpanKind = _SpanKind
__all__ = ['SpanKind']
INSTRUMENTING_MODULE_NAME = 'a2a-python-sdk'
INSTRUMENTING_MODULE_VERSION = '1.0.0'

logger = logging.getLogger(__name__)


def trace_function(
    func=None,
    *,
    span_name=None,
    kind=SpanKind.INTERNAL,
    attributes=None,
    attribute_extractor=None,
):
    """A decorator to automatically trace a function call with OpenTelemetry.

    This decorator can be used to wrap both sync and async functions.
    When applied, it creates a new span for each call to the decorated function.
    The span will record the execution time, status (OK or ERROR), and any
    exceptions that occur.

    It can be used in two ways:

    1. As a direct decorator: `@trace_function`
    2. As a decorator factory to provide arguments: `@trace_function(span_name="custom.name")`

    Args:
        func (callable, optional): The function to be decorated. If None,
            the decorator returns a partial function, allowing it to be called
            with arguments. Defaults to None.
        span_name (str, optional): Custom name for the span. If None,
            it defaults to ``f'{func.__module__}.{func.__name__}'``.
            Defaults to None.
        kind (SpanKind, optional): The ``opentelemetry.trace.SpanKind`` for the
            created span. Defaults to ``SpanKind.INTERNAL``.
        attributes (dict, optional): A dictionary of static attributes to be
            set on the span. Keys are attribute names (str) and values are
            the corresponding attribute values. Defaults to None.
        attribute_extractor (callable, optional): A function that can be used
            to dynamically extract and set attributes on the span.
            It is called within a ``finally`` block, ensuring it runs even if
            the decorated function raises an exception.
            The function signature should be:
            ``attribute_extractor(span, args, kwargs, result, exception)``
            where:
                - ``span`` : the OpenTelemetry ``Span`` object.
                - ``args`` : a tuple of positional arguments passed
                - ``kwargs`` : a dictionary of keyword arguments passed
                - ``result`` : return value (None if an exception occurred)
                - ``exception`` : exception object if raised (None otherwise).
            Any exception raised by the ``attribute_extractor`` itself will be
            caught and logged. Defaults to None.

    Returns:
        callable: The wrapped function that includes tracing, or a partial
            decorator if ``func`` is None.
    """
    if func is None:
        return functools.partial(
            trace_function,
            span_name=span_name,
            kind=kind,
            attributes=attributes,
            attribute_extractor=attribute_extractor,
        )

    actual_span_name = span_name or f'{func.__module__}.{func.__name__}'

    is_async_func = inspect.iscoroutinefunction(func)

    logger.debug(
        f'Start tracing for {actual_span_name}, is_async_func {is_async_func}'
    )

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> any:
        """Async Wrapper for the decorator."""
        logger.debug('Start async tracer')
        tracer = trace.get_tracer(
            INSTRUMENTING_MODULE_NAME, INSTRUMENTING_MODULE_VERSION
        )
        with tracer.start_as_current_span(actual_span_name, kind=kind) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, v)

            result = None
            exception = None

            try:
                # Async wrapper, await for the function call to complete.
                result = await func(*args, **kwargs)
                span.set_status(StatusCode.OK)
                return result

            except Exception as e:
                exception = e
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, description=str(e))
                raise
            finally:
                if attribute_extractor:
                    try:
                        attribute_extractor(
                            span, args, kwargs, result, exception
                        )
                    except Exception as attr_e:
                        logger.error(
                            f'attribute_extractor error in span {actual_span_name}: {attr_e}'
                        )

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        """Sync Wrapper for the decorator."""
        tracer = trace.get_tracer(INSTRUMENTING_MODULE_NAME)
        with tracer.start_as_current_span(actual_span_name, kind=kind) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, v)

            result = None
            exception = None

            try:
                # Sync wrapper, execute the function call.
                result = func(*args, **kwargs)
                span.set_status(StatusCode.OK)
                return result

            except Exception as e:
                exception = e
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, description=str(e))
                raise
            finally:
                if attribute_extractor:
                    try:
                        attribute_extractor(
                            span, args, kwargs, result, exception
                        )
                    except Exception as attr_e:
                        logger.error(
                            f'attribute_extractor error in span {actual_span_name}: {attr_e}'
                        )

    return async_wrapper if is_async_func else sync_wrapper


def trace_class(
    include_list: list[str] | None = None,
    exclude_list: list[str] | None = None,
    kind=SpanKind.INTERNAL,
):
    """A class decorator to automatically trace specified methods of a class.

    This decorator iterates over the methods of a class and applies the
    `trace_function` decorator to them, based on the `include_list` and
    `exclude_list` criteria. Methods starting or ending with double underscores
    (dunder methods, e.g., `__init__`, `__call__`) are always excluded by default.

    Args:
        include_list (list[str], optional): A list of method names to
            explicitly include for tracing. If provided, only methods in this
            list (that are not dunder methods) will be traced.
            Defaults to None (trace all non-dunder methods).
        exclude_list (list[str], optional): A list of method names to exclude
            from tracing. This is only considered if `include_list` is not
            provided. Dunder methods are implicitly excluded.
            Defaults to an empty list.
        kind (SpanKind, optional): The `opentelemetry.trace.SpanKind` for the
            created spans on the methods. Defaults to `SpanKind.INTERNAL`.

    Returns:
        callable: A decorator function that, when applied to a class,
                  modifies the class to wrap its specified methods with tracing.

    Example:
        To trace all methods except 'internal_method':
        ```python
        @trace_class(exclude_list=['internal_method'])
        class MyService:
            def public_api(self):
                pass

            def internal_method(self):
                pass
        ```

        To trace only 'method_one' and 'method_two':
        ```python
        @trace_class(include_list=['method_one', 'method_two'])
        class AnotherService:
            def method_one(self):
                pass

            def method_two(self):
                pass

            def not_traced_method(self):
                pass
        ```
    """
    logger.debug(f'Trace all class {include_list}, {exclude_list}')
    exclude_list = exclude_list or []

    def decorator(cls):
        all_methods = {}
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            # Skip Dunders
            if name.startswith('__') and name.endswith('__'):
                continue

            # Skip if include list is defined but the method not included.
            if include_list and name not in include_list:
                continue
            # Skip if include list is not defined but the method is in excludes.
            if not include_list and name in exclude_list:
                continue

            all_methods[name] = method
            span_name = f'{cls.__module__}.{cls.__name__}.{name}'
            # Set the decorator on the method.
            setattr(
                cls,
                name,
                trace_function(span_name=span_name, kind=kind)(method),
            )
        return cls

    return decorator
````

## File: tests/server/events/test_event_consumer.py
````python
import asyncio

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from a2a.server.events.event_consumer import EventConsumer
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    A2AError,
    Artifact,
    InternalError,
    JSONRPCError,
    Message,
    Part,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.utils.errors import ServerError


MINIMAL_TASK: dict[str, Any] = {
    'id': '123',
    'contextId': 'session-xyz',
    'status': {'state': 'submitted'},
    'kind': 'task',
}

MESSAGE_PAYLOAD: dict[str, Any] = {
    'role': 'agent',
    'parts': [{'text': 'test message'}],
    'messageId': '111',
}


@pytest.fixture
def mock_event_queue():
    return AsyncMock(spec=EventQueue)


@pytest.fixture
def event_consumer(mock_event_queue: EventQueue):
    return EventConsumer(queue=mock_event_queue)


@pytest.mark.asyncio
async def test_consume_one_task_event(
    event_consumer: MagicMock,
    mock_event_queue: MagicMock,
):
    task_event = Task(**MINIMAL_TASK)
    mock_event_queue.dequeue_event.return_value = task_event
    result = await event_consumer.consume_one()
    assert result == task_event
    mock_event_queue.task_done.assert_called_once()


@pytest.mark.asyncio
async def test_consume_one_message_event(
    event_consumer: MagicMock,
    mock_event_queue: MagicMock,
):
    message_event = Message(**MESSAGE_PAYLOAD)
    mock_event_queue.dequeue_event.return_value = message_event
    result = await event_consumer.consume_one()
    assert result == message_event
    mock_event_queue.task_done.assert_called_once()


@pytest.mark.asyncio
async def test_consume_one_a2a_error_event(
    event_consumer: MagicMock,
    mock_event_queue: MagicMock,
):
    error_event = A2AError(InternalError())
    mock_event_queue.dequeue_event.return_value = error_event
    result = await event_consumer.consume_one()
    assert result == error_event
    mock_event_queue.task_done.assert_called_once()


@pytest.mark.asyncio
async def test_consume_one_jsonrpc_error_event(
    event_consumer: MagicMock,
    mock_event_queue: MagicMock,
):
    error_event = JSONRPCError(code=123, message='Some Error')
    mock_event_queue.dequeue_event.return_value = error_event
    result = await event_consumer.consume_one()
    assert result == error_event
    mock_event_queue.task_done.assert_called_once()


@pytest.mark.asyncio
async def test_consume_one_queue_empty(
    event_consumer: MagicMock,
    mock_event_queue: MagicMock,
):
    mock_event_queue.dequeue_event.side_effect = asyncio.QueueEmpty
    try:
        result = await event_consumer.consume_one()
        assert result is not None
    except ServerError:
        pass
    mock_event_queue.task_done.assert_not_called()


@pytest.mark.asyncio
async def test_consume_all_multiple_events(
    event_consumer: MagicMock,
    mock_event_queue: MagicMock,
):
    events: list[Any] = [
        Task(**MINIMAL_TASK),
        TaskArtifactUpdateEvent(
            taskId='task_123',
            contextId='session-xyz',
            artifact=Artifact(
                artifactId='11', parts=[Part(TextPart(text='text'))]
            ),
        ),
        TaskStatusUpdateEvent(
            taskId='task_123',
            contextId='session-xyz',
            status=TaskStatus(state=TaskState.working),
            final=True,
        ),
    ]
    cursor = 0

    async def mock_dequeue() -> Any:
        nonlocal cursor
        if cursor < len(events):
            event = events[cursor]
            cursor += 1
            return event

    mock_event_queue.dequeue_event = mock_dequeue
    consumed_events: list[Any] = []
    async for event in event_consumer.consume_all():
        consumed_events.append(event)
    assert len(consumed_events) == 3
    assert consumed_events[0] == events[0]
    assert consumed_events[1] == events[1]
    assert consumed_events[2] == events[2]
    assert mock_event_queue.task_done.call_count == 3


@pytest.mark.asyncio
async def test_consume_until_message(
    event_consumer: MagicMock,
    mock_event_queue: MagicMock,
):
    events: list[Any] = [
        Task(**MINIMAL_TASK),
        TaskArtifactUpdateEvent(
            taskId='task_123',
            contextId='session-xyz',
            artifact=Artifact(
                artifactId='11', parts=[Part(TextPart(text='text'))]
            ),
        ),
        Message(**MESSAGE_PAYLOAD),
        TaskStatusUpdateEvent(
            taskId='task_123',
            contextId='session-xyz',
            status=TaskStatus(state=TaskState.working),
            final=True,
        ),
    ]
    cursor = 0

    async def mock_dequeue() -> Any:
        nonlocal cursor
        if cursor < len(events):
            event = events[cursor]
            cursor += 1
            return event

    mock_event_queue.dequeue_event = mock_dequeue
    consumed_events: list[Any] = []
    async for event in event_consumer.consume_all():
        consumed_events.append(event)
    assert len(consumed_events) == 3
    assert consumed_events[0] == events[0]
    assert consumed_events[1] == events[1]
    assert consumed_events[2] == events[2]
    assert mock_event_queue.task_done.call_count == 3


@pytest.mark.asyncio
async def test_consume_message_events(
    event_consumer: MagicMock,
    mock_event_queue: MagicMock,
):
    events = [
        Message(**MESSAGE_PAYLOAD),
        Message(**MESSAGE_PAYLOAD, final=True),
    ]
    cursor = 0

    async def mock_dequeue() -> Any:
        nonlocal cursor
        if cursor < len(events):
            event = events[cursor]
            cursor += 1
            return event

    mock_event_queue.dequeue_event = mock_dequeue
    consumed_events: list[Any] = []
    async for event in event_consumer.consume_all():
        consumed_events.append(event)
    # Upon first Message the stream is closed.
    assert len(consumed_events) == 1
    assert consumed_events[0] == events[0]
    assert mock_event_queue.task_done.call_count == 1
````

## File: tests/server/tasks/test_inmemory_task_store.py
````python
from typing import Any

import pytest

from a2a.server.tasks import InMemoryTaskStore
from a2a.types import Task


MINIMAL_TASK: dict[str, Any] = {
    'id': 'task-abc',
    'contextId': 'session-xyz',
    'status': {'state': 'submitted'},
    'kind': 'task',
}


@pytest.mark.asyncio
async def test_in_memory_task_store_save_and_get() -> None:
    """Test saving and retrieving a task from the in-memory store."""
    store = InMemoryTaskStore()
    task = Task(**MINIMAL_TASK)
    await store.save(task)
    retrieved_task = await store.get(MINIMAL_TASK['id'])
    assert retrieved_task == task


@pytest.mark.asyncio
async def test_in_memory_task_store_get_nonexistent() -> None:
    """Test retrieving a nonexistent task."""
    store = InMemoryTaskStore()
    retrieved_task = await store.get('nonexistent')
    assert retrieved_task is None


@pytest.mark.asyncio
async def test_in_memory_task_store_delete() -> None:
    """Test deleting a task from the store."""
    store = InMemoryTaskStore()
    task = Task(**MINIMAL_TASK)
    await store.save(task)
    await store.delete(MINIMAL_TASK['id'])
    retrieved_task = await store.get(MINIMAL_TASK['id'])
    assert retrieved_task is None


@pytest.mark.asyncio
async def test_in_memory_task_store_delete_nonexistent() -> None:
    """Test deleting a nonexistent task."""
    store = InMemoryTaskStore()
    await store.delete('nonexistent')
````

## File: tests/server/tasks/test_task_manager.py
````python
from typing import Any
from unittest.mock import AsyncMock

import pytest

from a2a.server.tasks import TaskManager
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)


MINIMAL_TASK: dict[str, Any] = {
    'id': 'task-abc',
    'contextId': 'session-xyz',
    'status': {'state': 'submitted'},
    'kind': 'task',
}


@pytest.fixture
def mock_task_store() -> AsyncMock:
    """Fixture for a mock TaskStore."""
    return AsyncMock()


@pytest.fixture
def task_manager(mock_task_store: AsyncMock) -> TaskManager:
    """Fixture for a TaskManager with a mock TaskStore."""
    return TaskManager(
        task_id=MINIMAL_TASK['id'],
        context_id=MINIMAL_TASK['contextId'],
        task_store=mock_task_store,
        initial_message=None,
    )


@pytest.mark.asyncio
async def test_get_task_existing(
    task_manager: TaskManager, mock_task_store: AsyncMock
) -> None:
    """Test getting an existing task."""
    expected_task = Task(**MINIMAL_TASK)
    mock_task_store.get.return_value = expected_task
    retrieved_task = await task_manager.get_task()
    assert retrieved_task == expected_task
    mock_task_store.get.assert_called_once_with(MINIMAL_TASK['id'])


@pytest.mark.asyncio
async def test_get_task_nonexistent(
    task_manager: TaskManager, mock_task_store: AsyncMock
) -> None:
    """Test getting a nonexistent task."""
    mock_task_store.get.return_value = None
    retrieved_task = await task_manager.get_task()
    assert retrieved_task is None
    mock_task_store.get.assert_called_once_with(MINIMAL_TASK['id'])


@pytest.mark.asyncio
async def test_save_task_event_new_task(
    task_manager: TaskManager, mock_task_store: AsyncMock
) -> None:
    """Test saving a new task."""
    task = Task(**MINIMAL_TASK)
    await task_manager.save_task_event(task)
    mock_task_store.save.assert_called_once_with(task)


@pytest.mark.asyncio
async def test_save_task_event_status_update(
    task_manager: TaskManager, mock_task_store: AsyncMock
) -> None:
    """Test saving a status update for an existing task."""
    initial_task = Task(**MINIMAL_TASK)
    mock_task_store.get.return_value = initial_task
    new_status = TaskStatus(
        state=TaskState.working,
        message=Message(
            role=Role.agent,
            parts=[Part(TextPart(text='content'))],
            messageId='message-id',
        ),
    )
    event = TaskStatusUpdateEvent(
        taskId=MINIMAL_TASK['id'],
        contextId=MINIMAL_TASK['contextId'],
        status=new_status,
        final=False,
    )
    await task_manager.save_task_event(event)
    updated_task = initial_task
    updated_task.status = new_status
    mock_task_store.save.assert_called_once_with(updated_task)


@pytest.mark.asyncio
async def test_save_task_event_artifact_update(
    task_manager: TaskManager, mock_task_store: AsyncMock
) -> None:
    """Test saving an artifact update for an existing task."""
    initial_task = Task(**MINIMAL_TASK)
    mock_task_store.get.return_value = initial_task
    new_artifact = Artifact(
        artifactId='artifact-id',
        name='artifact1',
        parts=[Part(TextPart(text='content'))],
    )
    event = TaskArtifactUpdateEvent(
        taskId=MINIMAL_TASK['id'],
        contextId=MINIMAL_TASK['contextId'],
        artifact=new_artifact,
    )
    await task_manager.save_task_event(event)
    updated_task = initial_task
    updated_task.artifacts = [new_artifact]
    mock_task_store.save.assert_called_once_with(updated_task)


@pytest.mark.asyncio
async def test_ensure_task_existing(
    task_manager: TaskManager, mock_task_store: AsyncMock
) -> None:
    """Test ensuring an existing task."""
    expected_task = Task(**MINIMAL_TASK)
    mock_task_store.get.return_value = expected_task
    event = TaskStatusUpdateEvent(
        taskId=MINIMAL_TASK['id'],
        contextId=MINIMAL_TASK['contextId'],
        status=TaskStatus(state=TaskState.working),
        final=False,
    )
    retrieved_task = await task_manager.ensure_task(event)
    assert retrieved_task == expected_task
    mock_task_store.get.assert_called_once_with(MINIMAL_TASK['id'])


@pytest.mark.asyncio
async def test_ensure_task_nonexistent(
    mock_task_store: AsyncMock,
) -> None:
    """Test ensuring a nonexistent task (creates a new one)."""
    mock_task_store.get.return_value = None
    task_manager_without_id = TaskManager(
        task_id=None,
        context_id=None,
        task_store=mock_task_store,
        initial_message=None,
    )
    event = TaskStatusUpdateEvent(
        taskId='new-task',
        contextId='some-context',
        status=TaskStatus(state=TaskState.submitted),
        final=False,
    )
    new_task = await task_manager_without_id.ensure_task(event)
    assert new_task.id == 'new-task'
    assert new_task.contextId == 'some-context'
    assert new_task.status.state == TaskState.submitted
    mock_task_store.save.assert_called_once_with(new_task)
    assert task_manager_without_id.task_id == 'new-task'
    assert task_manager_without_id.context_id == 'some-context'


def test_init_task_obj(task_manager: TaskManager) -> None:
    """Test initializing a new task object."""
    new_task = task_manager._init_task_obj('new-task', 'new-context')  # type: ignore
    assert new_task.id == 'new-task'
    assert new_task.contextId == 'new-context'
    assert new_task.status.state == TaskState.submitted
    assert new_task.history == []


@pytest.mark.asyncio
async def test_save_task(
    task_manager: TaskManager, mock_task_store: AsyncMock
) -> None:
    """Test saving a task."""
    task = Task(**MINIMAL_TASK)
    await task_manager._save_task(task)  # type: ignore
    mock_task_store.save.assert_called_once_with(task)


@pytest.mark.asyncio
async def test_save_task_event_new_task_no_task_id(
    mock_task_store: AsyncMock,
) -> None:
    """Test saving a task event without task id in TaskManager."""
    task_manager_without_id = TaskManager(
        task_id=None,
        context_id=None,
        task_store=mock_task_store,
        initial_message=None,
    )
    task_data: dict[str, Any] = {
        'id': 'new-task-id',
        'contextId': 'some-context',
        'status': {'state': 'working'},
        'kind': 'task',
    }
    task = Task(**task_data)
    await task_manager_without_id.save_task_event(task)
    mock_task_store.save.assert_called_once_with(task)
    assert task_manager_without_id.task_id == 'new-task-id'
    assert task_manager_without_id.context_id == 'some-context'
    # initial submit should be updated to working
    assert task.status.state == TaskState.working


@pytest.mark.asyncio
async def test_get_task_no_task_id(
    mock_task_store: AsyncMock,
) -> None:
    """Test getting a task when task_id is not set in TaskManager."""
    task_manager_without_id = TaskManager(
        task_id=None,
        context_id='some-context',
        task_store=mock_task_store,
        initial_message=None,
    )
    retrieved_task = await task_manager_without_id.get_task()
    assert retrieved_task is None
    mock_task_store.get.assert_not_called()


@pytest.mark.asyncio
async def test_save_task_event_no_task_existing(
    mock_task_store: AsyncMock,
) -> None:
    """Test saving an event when no task exists and task_id is not set."""
    task_manager_without_id = TaskManager(
        task_id=None,
        context_id=None,
        task_store=mock_task_store,
        initial_message=None,
    )
    mock_task_store.get.return_value = None
    event = TaskStatusUpdateEvent(
        taskId='event-task-id',
        contextId='some-context',
        status=TaskStatus(state=TaskState.completed),
        final=True,
    )
    await task_manager_without_id.save_task_event(event)
    # Check if a new task was created and saved
    call_args = mock_task_store.save.call_args
    assert call_args is not None
    saved_task = call_args[0][0]
    assert saved_task.id == 'event-task-id'
    assert saved_task.contextId == 'some-context'
    assert saved_task.status.state == TaskState.completed
    assert task_manager_without_id.task_id == 'event-task-id'
    assert task_manager_without_id.context_id == 'some-context'
````

## File: tests/server/test_integration.py
````python
import asyncio
from typing import Any
from unittest import mock

import pytest
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from a2a.server.apps.starlette_app import A2AStarletteApplication
from a2a.types import (AgentCapabilities, AgentCard, Artifact, DataPart,
                       InternalError, InvalidRequestError, JSONParseError,
                       Part, PushNotificationConfig, Task,
                       TaskArtifactUpdateEvent, TaskPushNotificationConfig,
                       TaskState, TaskStatus, TextPart,
                       UnsupportedOperationError)
from a2a.utils.errors import MethodNotImplementedError

# === TEST SETUP ===

MINIMAL_AGENT_SKILL: dict[str, Any] = {
    'id': 'skill-123',
    'name': 'Recipe Finder',
    'description': 'Finds recipes',
    'tags': ['cooking'],
}

MINIMAL_AGENT_AUTH: dict[str, Any] = {'schemes': ['Bearer']}

AGENT_CAPS = AgentCapabilities(
    pushNotifications=True, stateTransitionHistory=False, streaming=True
)

MINIMAL_AGENT_CARD: dict[str, Any] = {
    'authentication': MINIMAL_AGENT_AUTH,
    'capabilities': AGENT_CAPS,  # AgentCapabilities is required but can be empty
    'defaultInputModes': ['text/plain'],
    'defaultOutputModes': ['application/json'],
    'description': 'Test Agent',
    'name': 'TestAgent',
    'skills': [MINIMAL_AGENT_SKILL],
    'url': 'http://example.com/agent',
    'version': '1.0',
}

EXTENDED_AGENT_CARD_DATA: dict[str, Any] = {
    **MINIMAL_AGENT_CARD,
    'name': 'TestAgent Extended',
    'description': 'Test Agent with more details',
    'skills': [
        MINIMAL_AGENT_SKILL,
        {
            'id': 'skill-extended',
            'name': 'Extended Skill',
            'description': 'Does more things',
            'tags': ['extended'],
        },
    ],
}
TEXT_PART_DATA: dict[str, Any] = {'kind': 'text', 'text': 'Hello'}

DATA_PART_DATA: dict[str, Any] = {'kind': 'data', 'data': {'key': 'value'}}

MINIMAL_MESSAGE_USER: dict[str, Any] = {
    'role': 'user',
    'parts': [TEXT_PART_DATA],
    'messageId': 'msg-123',
    'kind': 'message',
}

MINIMAL_TASK_STATUS: dict[str, Any] = {'state': 'submitted'}

FULL_TASK_STATUS: dict[str, Any] = {
    'state': 'working',
    'message': MINIMAL_MESSAGE_USER,
    'timestamp': '2023-10-27T10:00:00Z',
}


@pytest.fixture
def agent_card():
    return AgentCard(**MINIMAL_AGENT_CARD)


@pytest.fixture
def extended_agent_card_fixture():
    return AgentCard(**EXTENDED_AGENT_CARD_DATA)


@pytest.fixture
def handler():
    handler = mock.AsyncMock()
    handler.on_message_send = mock.AsyncMock()
    handler.on_cancel_task = mock.AsyncMock()
    handler.on_get_task = mock.AsyncMock()
    handler.set_push_notification = mock.AsyncMock()
    handler.get_push_notification = mock.AsyncMock()
    handler.on_message_send_stream = mock.Mock()
    handler.on_resubscribe_to_task = mock.Mock()
    return handler


@pytest.fixture
def app(agent_card: AgentCard, handler: mock.AsyncMock):
    return A2AStarletteApplication(agent_card, handler)


@pytest.fixture
def client(app: A2AStarletteApplication):
    """Create a test client with the app."""
    return TestClient(app.build())


# === BASIC FUNCTIONALITY TESTS ===


def test_agent_card_endpoint(client: TestClient, agent_card: AgentCard):
    """Test the agent card endpoint returns expected data."""
    response = client.get('/.well-known/agent.json')
    assert response.status_code == 200
    data = response.json()
    assert data['name'] == agent_card.name
    assert data['version'] == agent_card.version
    assert 'streaming' in data['capabilities']


def test_authenticated_extended_agent_card_endpoint_not_supported(
    agent_card: AgentCard, handler: mock.AsyncMock
):
    """Test extended card endpoint returns 404 if not supported by main card."""
    # Ensure supportsAuthenticatedExtendedCard is False or None
    agent_card.supportsAuthenticatedExtendedCard = False
    app_instance = A2AStarletteApplication(agent_card, handler)
    # The route should not even be added if supportsAuthenticatedExtendedCard is false
    # So, building the app and trying to hit it should result in 404 from Starlette itself
    client = TestClient(app_instance.build())
    response = client.get('/agent/authenticatedExtendedCard')
    assert response.status_code == 404 # Starlette's default for no route


def test_authenticated_extended_agent_card_endpoint_supported_with_specific_extended_card(
    agent_card: AgentCard,
    extended_agent_card_fixture: AgentCard,
    handler: mock.AsyncMock,
):
    """Test extended card endpoint returns the specific extended card when provided."""
    agent_card.supportsAuthenticatedExtendedCard = True # Main card must support it
    app_instance = A2AStarletteApplication(
        agent_card, handler, extended_agent_card=extended_agent_card_fixture
    )
    client = TestClient(app_instance.build())

    response = client.get('/agent/authenticatedExtendedCard')
    assert response.status_code == 200
    data = response.json()
    # Verify it's the extended card's data
    assert data['name'] == extended_agent_card_fixture.name
    assert data['version'] == extended_agent_card_fixture.version
    assert len(data['skills']) == len(extended_agent_card_fixture.skills)
    assert any(
        skill['id'] == 'skill-extended' for skill in data['skills']
    ), "Extended skill not found in served card"



def test_agent_card_custom_url(
    app: A2AStarletteApplication, agent_card: AgentCard
):
    """Test the agent card endpoint with a custom URL."""
    client = TestClient(app.build(agent_card_url='/my-agent'))
    response = client.get('/my-agent')
    assert response.status_code == 200
    data = response.json()
    assert data['name'] == agent_card.name


def test_rpc_endpoint_custom_url(
    app: A2AStarletteApplication, handler: mock.AsyncMock
):
    """Test the RPC endpoint with a custom URL."""
    # Provide a valid Task object as the return value
    task_status = TaskStatus(**MINIMAL_TASK_STATUS)
    task = Task(
        id='task1', contextId='ctx1', state='completed', status=task_status
    )
    handler.on_get_task.return_value = task
    client = TestClient(app.build(rpc_url='/api/rpc'))
    response = client.post(
        '/api/rpc',
        json={
            'jsonrpc': '2.0',
            'id': '123',
            'method': 'tasks/get',
            'params': {'id': 'task1'},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data['result']['id'] == 'task1'


def test_build_with_extra_routes(
    app: A2AStarletteApplication, agent_card: AgentCard
):
    """Test building the app with additional routes."""

    def custom_handler(request):
        return JSONResponse({'message': 'Hello'})

    extra_route = Route('/hello', custom_handler, methods=['GET'])
    test_app = app.build(routes=[extra_route])
    client = TestClient(test_app)

    # Test the added route
    response = client.get('/hello')
    assert response.status_code == 200
    assert response.json() == {'message': 'Hello'}

    # Ensure default routes still work
    response = client.get('/.well-known/agent.json')
    assert response.status_code == 200
    data = response.json()
    assert data['name'] == agent_card.name


# === REQUEST METHODS TESTS ===


def test_send_message(client: TestClient, handler: mock.AsyncMock):
    """Test sending a message."""
    # Prepare mock response
    task_status = TaskStatus(**MINIMAL_TASK_STATUS)
    mock_task = Task(
        id='task1',
        contextId='session-xyz',
        state='completed',
        status=task_status,
    )
    handler.on_message_send.return_value = mock_task

    # Send request
    response = client.post(
        '/',
        json={
            'jsonrpc': '2.0',
            'id': '123',
            'method': 'message/send',
            'params': {
                'message': {
                    'role': 'agent',
                    'parts': [{'kind': 'text', 'text': 'Hello'}],
                    'messageId': '111',
                    'kind': 'message',
                    'taskId': 'task1',
                    'contextId': 'session-xyz',
                }
            },
        },
    )

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert 'result' in data
    assert data['result']['id'] == 'task1'
    assert data['result']['status']['state'] == 'submitted'

    # Verify handler was called
    handler.on_message_send.assert_awaited_once()


def test_cancel_task(client: TestClient, handler: mock.AsyncMock):
    """Test cancelling a task."""
    # Setup mock response
    task_status = TaskStatus(**MINIMAL_TASK_STATUS)
    task_status.state = TaskState.canceled  # 'cancelled' #
    task = Task(
        id='task1', contextId='ctx1', state='cancelled', status=task_status
    )
    handler.on_cancel_task.return_value = task

    # Send request
    response = client.post(
        '/',
        json={
            'jsonrpc': '2.0',
            'id': '123',
            'method': 'tasks/cancel',
            'params': {'id': 'task1'},
        },
    )

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data['result']['id'] == 'task1'
    assert data['result']['status']['state'] == 'canceled'

    # Verify handler was called
    handler.on_cancel_task.assert_awaited_once()


def test_get_task(client: TestClient, handler: mock.AsyncMock):
    """Test getting a task."""
    # Setup mock response
    task_status = TaskStatus(**MINIMAL_TASK_STATUS)
    task = Task(
        id='task1', contextId='ctx1', state='completed', status=task_status
    )
    handler.on_get_task.return_value = task  # JSONRPCResponse(root=task)

    # Send request
    response = client.post(
        '/',
        json={
            'jsonrpc': '2.0',
            'id': '123',
            'method': 'tasks/get',
            'params': {'id': 'task1'},
        },
    )

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data['result']['id'] == 'task1'

    # Verify handler was called
    handler.on_get_task.assert_awaited_once()


def test_set_push_notification_config(
    client: TestClient, handler: mock.AsyncMock
):
    """Test setting push notification configuration."""
    # Setup mock response
    task_push_config = TaskPushNotificationConfig(
        taskId='t2',
        pushNotificationConfig=PushNotificationConfig(
            url='https://example.com', token='secret-token'
        ),
    )
    handler.on_set_task_push_notification_config.return_value = task_push_config

    # Send request
    response = client.post(
        '/',
        json={
            'jsonrpc': '2.0',
            'id': '123',
            'method': 'tasks/pushNotificationConfig/set',
            'params': {
                'taskId': 't2',
                'pushNotificationConfig': {
                    'url': 'https://example.com',
                    'token': 'secret-token',
                },
            },
        },
    )

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data['result']['pushNotificationConfig']['token'] == 'secret-token'

    # Verify handler was called
    handler.on_set_task_push_notification_config.assert_awaited_once()


def test_get_push_notification_config(
    client: TestClient, handler: mock.AsyncMock
):
    """Test getting push notification configuration."""
    # Setup mock response
    task_push_config = TaskPushNotificationConfig(
        taskId='task1',
        pushNotificationConfig=PushNotificationConfig(
            url='https://example.com', token='secret-token'
        ),
    )

    handler.on_get_task_push_notification_config.return_value = task_push_config

    # Send request
    response = client.post(
        '/',
        json={
            'jsonrpc': '2.0',
            'id': '123',
            'method': 'tasks/pushNotificationConfig/get',
            'params': {'id': 'task1'},
        },
    )

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data['result']['pushNotificationConfig']['token'] == 'secret-token'

    # Verify handler was called
    handler.on_get_task_push_notification_config.assert_awaited_once()


# === STREAMING TESTS ===


@pytest.mark.asyncio
async def test_message_send_stream(
    app: A2AStarletteApplication, handler: mock.AsyncMock
) -> None:
    """Test streaming message sending."""

    # Setup mock streaming response
    async def stream_generator():
        for i in range(3):
            text_part = TextPart(**TEXT_PART_DATA)
            data_part = DataPart(**DATA_PART_DATA)
            artifact = Artifact(
                artifactId=f'artifact-{i}',
                name='result_data',
                parts=[Part(root=text_part), Part(root=data_part)],
            )
            last = [False, False, True]
            task_artifact_update_event_data: dict[str, Any] = {
                'artifact': artifact,
                'taskId': 'task_id',
                'contextId': 'session-xyz',
                'append': False,
                'lastChunk': last[i],
                'kind': 'artifact-update',
            }

            yield TaskArtifactUpdateEvent.model_validate(
                task_artifact_update_event_data
            )

    handler.on_message_send_stream.return_value = stream_generator()

    client = None
    try:
        # Create client
        client = TestClient(app.build(), raise_server_exceptions=False)
        # Send request
        with client.stream(
            'POST',
            '/',
            json={
                'jsonrpc': '2.0',
                'id': '123',
                'method': 'message/stream',
                'params': {
                    'message': {
                        'role': 'agent',
                        'parts': [{'kind': 'text', 'text': 'Hello'}],
                        'messageId': '111',
                        'kind': 'message',
                        'taskId': 'taskId',
                        'contextId': 'session-xyz',
                    }
                },
            },
        ) as response:
            # Verify response is a stream
            assert response.status_code == 200
            assert response.headers['content-type'].startswith(
                'text/event-stream'
            )

            # Read some content to verify streaming works
            content = b''
            event_count = 0

            for chunk in response.iter_bytes():
                content += chunk
                if b'data' in chunk:  # Naive check for SSE data lines
                    event_count += 1

            # Check content has event data (e.g., part of the first event)
            assert (
                b'"artifactId":"artifact-0"' in content
            )  # Check for the actual JSON payload
            assert (
                b'"artifactId":"artifact-1"' in content
            )  # Check for the actual JSON payload
            assert (
                b'"artifactId":"artifact-2"' in content
            )  # Check for the actual JSON payload
            assert event_count > 0
    finally:
        # Ensure the client is closed
        if client:
            client.close()
        # Allow event loop to process any pending callbacks
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_task_resubscription(
    app: A2AStarletteApplication, handler: mock.AsyncMock
) -> None:
    """Test task resubscription streaming."""

    # Setup mock streaming response
    async def stream_generator():
        for i in range(3):
            text_part = TextPart(**TEXT_PART_DATA)
            data_part = DataPart(**DATA_PART_DATA)
            artifact = Artifact(
                artifactId=f'artifact-{i}',
                name='result_data',
                parts=[Part(root=text_part), Part(root=data_part)],
            )
            last = [False, False, True]
            task_artifact_update_event_data: dict[str, Any] = {
                'artifact': artifact,
                'taskId': 'task_id',
                'contextId': 'session-xyz',
                'append': False,
                'lastChunk': last[i],
                'kind': 'artifact-update',
            }
            yield TaskArtifactUpdateEvent.model_validate(
                task_artifact_update_event_data
            )

    handler.on_resubscribe_to_task.return_value = stream_generator()

    # Create client
    client = TestClient(app.build(), raise_server_exceptions=False)

    try:
        # Send request using client.stream() context manager
        # Send request
        with client.stream(
            'POST',
            '/',
            json={
                'jsonrpc': '2.0',
                'id': '123',  # This ID is used in the success_event above
                'method': 'tasks/resubscribe',
                'params': {'id': 'task1'},
            },
        ) as response:
            # Verify response is a stream
            assert response.status_code == 200
            assert (
                response.headers['content-type']
                == 'text/event-stream; charset=utf-8'
            )

            # Read some content to verify streaming works
            content = b''
            event_count = 0
            for chunk in response.iter_bytes():
                content += chunk
                # A more robust check would be to parse each SSE event
                if b'data:' in chunk:  # Naive check for SSE data lines
                    event_count += 1
                if (
                    event_count >= 1 and len(content) > 20
                ):  # Ensure we've processed at least one event
                    break

            # Check content has event data (e.g., part of the first event)
            assert (
                b'"artifactId":"artifact-0"' in content
            )  # Check for the actual JSON payload
            assert (
                b'"artifactId":"artifact-1"' in content
            )  # Check for the actual JSON payload
            assert (
                b'"artifactId":"artifact-2"' in content
            )  # Check for the actual JSON payload
            assert event_count > 0
    finally:
        # Ensure the client is closed
        if client:
            client.close()
        # Allow event loop to process any pending callbacks
        await asyncio.sleep(0.1)


# === ERROR HANDLING TESTS ===


def test_invalid_json(client: TestClient):
    """Test handling invalid JSON."""
    response = client.post('/', content=b'This is not JSON')  # Use bytes
    assert response.status_code == 200  # JSON-RPC errors still return 200
    data = response.json()
    assert 'error' in data
    assert data['error']['code'] == JSONParseError().code


def test_invalid_request_structure(client: TestClient):
    """Test handling an invalid request structure."""
    response = client.post(
        '/',
        json={
            # Missing required fields
            'id': '123'
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert 'error' in data
    assert data['error']['code'] == InvalidRequestError().code


def test_method_not_implemented(client: TestClient, handler: mock.AsyncMock):
    """Test handling MethodNotImplementedError."""
    handler.on_get_task.side_effect = MethodNotImplementedError()

    response = client.post(
        '/',
        json={
            'jsonrpc': '2.0',
            'id': '123',
            'method': 'tasks/get',
            'params': {'id': 'task1'},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert 'error' in data
    assert data['error']['code'] == UnsupportedOperationError().code


def test_unknown_method(client: TestClient):
    """Test handling unknown method."""
    response = client.post(
        '/',
        json={
            'jsonrpc': '2.0',
            'id': '123',
            'method': 'unknown/method',
            'params': {},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert 'error' in data
    # This should produce an UnsupportedOperationError error code
    assert data['error']['code'] == InvalidRequestError().code


def test_validation_error(client: TestClient):
    """Test handling validation error."""
    # Missing required fields in the message
    response = client.post(
        '/',
        json={
            'jsonrpc': '2.0',
            'id': '123',
            'method': 'messages/send',
            'params': {
                'message': {
                    # Missing required fields
                    'text': 'Hello'
                }
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert 'error' in data
    assert data['error']['code'] == InvalidRequestError().code


def test_unhandled_exception(client: TestClient, handler: mock.AsyncMock):
    """Test handling unhandled exception."""
    handler.on_get_task.side_effect = Exception('Unexpected error')

    response = client.post(
        '/',
        json={
            'jsonrpc': '2.0',
            'id': '123',
            'method': 'tasks/get',
            'params': {'id': 'task1'},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert 'error' in data
    assert data['error']['code'] == InternalError().code
    assert 'Unexpected error' in data['error']['message']


def test_get_method_to_rpc_endpoint(client: TestClient):
    """Test sending GET request to RPC endpoint."""
    response = client.get('/')
    # Should return 405 Method Not Allowed
    assert response.status_code == 405


def test_non_dict_json(client: TestClient):
    """Test handling JSON that's not a dict."""
    response = client.post('/', json=['not', 'a', 'dict'])
    assert response.status_code == 200
    data = response.json()
    assert 'error' in data
    assert data['error']['code'] == InvalidRequestError().code
````

## File: CONTRIBUTING.md
````markdown
# How to contribute

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

### Contributor Guide

You may follow these steps to contribute:

1. **Fork the official repository.** This will create a copy of the official repository in your own account.
2. **Sync the branches.** This will ensure that your copy of the repository is up-to-date with the latest changes from the official repository.
3. **Work on your forked repository's feature branch.** This is where you will make your changes to the code.
4. **Commit your updates on your forked repository's feature branch.** This will save your changes to your copy of the repository.
5. **Submit a pull request to the official repository's main branch.** This will request that your changes be merged into the official repository.
6. **Resolve any linting errors.** This will ensure that your changes are formatted correctly.

Here are some additional things to keep in mind during the process:

- **Test your changes.** Before you submit a pull request, make sure that your changes work as expected.
- **Be patient.** It may take some time for your pull request to be reviewed and merged.

---

## For Google Employees

Complete the following steps to register your GitHub account and be added as a contributor to this repository.

1. Register your GitHub account at [go/GitHub](http://go/github).
````

## File: development.md
````markdown
# Development

## Type generation from spec

```bash
uv run datamodel-codegen \
  --url https://raw.githubusercontent.com/google-a2a/A2A/refs/heads/main/specification/json/a2a.json \
  --input-file-type jsonschema \
  --output ./src/a2a/types.py \
  --target-python-version 3.10 \
  --output-model-type pydantic_v2.BaseModel \
  --disable-timestamp \
  --use-schema-description \
  --use-union-operator \
  --use-field-description \
  --use-default \
  --use-default-kwarg \
  --use-one-literal-as-default \
  --class-name A2A \
  --use-standard-collections
```
````

## File: src/a2a/client/client.py
````python
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

import httpx
from httpx_sse import SSEError, aconnect_sse
from pydantic import ValidationError

from a2a.client.errors import A2AClientHTTPError, A2AClientJSONError
from a2a.types import (AgentCard, CancelTaskRequest, CancelTaskResponse,
                       GetTaskPushNotificationConfigRequest,
                       GetTaskPushNotificationConfigResponse, GetTaskRequest,
                       GetTaskResponse, SendMessageRequest,
                       SendMessageResponse, SendStreamingMessageRequest,
                       SendStreamingMessageResponse,
                       SetTaskPushNotificationConfigRequest,
                       SetTaskPushNotificationConfigResponse)
from a2a.utils.telemetry import SpanKind, trace_class

logger = logging.getLogger(__name__)

class A2ACardResolver:
    """Agent Card resolver."""

    def __init__(
        self,
        httpx_client: httpx.AsyncClient,
        base_url: str,
        agent_card_path: str = '/.well-known/agent.json',
    ):
        """Initializes the A2ACardResolver.

        Args:
            httpx_client: An async HTTP client instance (e.g., httpx.AsyncClient).
            base_url: The base URL of the agent's host.
            agent_card_path: The path to the agent card endpoint, relative to the base URL.
        """
        self.base_url = base_url.rstrip('/')
        self.agent_card_path = agent_card_path.lstrip('/')
        self.httpx_client = httpx_client

    async def get_agent_card(
        self,
        relative_card_path: str | None = None,
        http_kwargs: dict[str, Any] | None = None,
    ) -> AgentCard:
        """Fetches an agent card from a specified path relative to the base_url.

        If relative_card_path is None, it defaults to the resolver's configured
        agent_card_path (for the public agent card).

        Args:
            relative_card_path: Optional path to the agent card endpoint,
                relative to the base URL. If None, uses the default public
                agent card path.
            http_kwargs: Optional dictionary of keyword arguments to pass to the
                underlying httpx.get request.

        Returns:
            An `AgentCard` object representing the agent's capabilities.

        Raises:
            A2AClientHTTPError: If an HTTP error occurs during the request.
            A2AClientJSONError: If the response body cannot be decoded as JSON
                or validated against the AgentCard schema.
        """
        if relative_card_path is None:
            # Use the default public agent card path configured during initialization
            path_segment = self.agent_card_path
        else:
            path_segment = relative_card_path.lstrip('/')

        target_url = f'{self.base_url}/{path_segment}'

        try:
            response = await self.httpx_client.get(
                target_url,
                **(http_kwargs or {}),
            )
            response.raise_for_status()
            agent_card_data = response.json()
            logger.info(
                'Successfully fetched agent card data from %s: %s',
                target_url,
                agent_card_data,
            )
            agent_card = AgentCard.model_validate(agent_card_data)
        except httpx.HTTPStatusError as e:
            raise A2AClientHTTPError(
                e.response.status_code,
                f'Failed to fetch agent card from {target_url}: {e}',
            ) from e
        except json.JSONDecodeError as e:
            raise A2AClientJSONError(
                f'Failed to parse JSON for agent card from {target_url}: {e}'
            ) from e
        except httpx.RequestError as e:
            raise A2AClientHTTPError(
                503,
                f'Network communication error fetching agent card from {target_url}: {e}',
            ) from e
        except ValidationError as e:  # Pydantic validation error
            raise A2AClientJSONError(
                f'Failed to validate agent card structure from {target_url}: {e.json()}'
            ) from e

        return agent_card


@trace_class(kind=SpanKind.CLIENT)
class A2AClient:
    """A2A Client for interacting with an A2A agent."""

    def __init__(
        self,
        httpx_client: httpx.AsyncClient,
        agent_card: AgentCard | None = None,
        url: str | None = None,
    ):
        """Initializes the A2AClient.

        Requires either an `AgentCard` or a direct `url` to the agent's RPC endpoint.

        Args:
            httpx_client: An async HTTP client instance (e.g., httpx.AsyncClient).
            agent_card: The agent card object. If provided, `url` is taken from `agent_card.url`.
            url: The direct URL to the agent's A2A RPC endpoint. Required if `agent_card` is None.

        Raises:
            ValueError: If neither `agent_card` nor `url` is provided.
        """
        if agent_card:
            self.url = agent_card.url
        elif url:
            self.url = url
        else:
            raise ValueError('Must provide either agent_card or url')

        self.httpx_client = httpx_client

    @staticmethod
    async def get_client_from_agent_card_url(
        httpx_client: httpx.AsyncClient,
        base_url: str,
        agent_card_path: str = '/.well-known/agent.json',
        http_kwargs: dict[str, Any] | None = None,
    ) -> 'A2AClient':
        """Fetches the public AgentCard and initializes an A2A client.

        This method will always fetch the public agent card. If an authenticated
        or extended agent card is required, the A2ACardResolver should be used
        directly to fetch the specific card, and then the A2AClient should be
        instantiated with it.

        Args:
            httpx_client: An async HTTP client instance (e.g., httpx.AsyncClient).
            base_url: The base URL of the agent's host.
            agent_card_path: The path to the agent card endpoint, relative to the base URL.
            http_kwargs: Optional dictionary of keyword arguments to pass to the
                underlying httpx.get request when fetching the agent card.
        Returns:
            An initialized `A2AClient` instance.

        Raises:
            A2AClientHTTPError: If an HTTP error occurs fetching the agent card.
            A2AClientJSONError: If the agent card response is invalid.
        """
        agent_card: AgentCard = await A2ACardResolver(
            httpx_client, base_url=base_url, agent_card_path=agent_card_path
        ).get_agent_card(http_kwargs=http_kwargs) # Fetches public card by default
        return A2AClient(httpx_client=httpx_client, agent_card=agent_card)

    async def send_message(
        self,
        request: SendMessageRequest,
        *,
        http_kwargs: dict[str, Any] | None = None,
    ) -> SendMessageResponse:
        """Sends a non-streaming message request to the agent.

        Args:
            request: The `SendMessageRequest` object containing the message and configuration.
            http_kwargs: Optional dictionary of keyword arguments to pass to the
                underlying httpx.post request.

        Returns:
            A `SendMessageResponse` object containing the agent's response (Task or Message) or an error.

        Raises:
            A2AClientHTTPError: If an HTTP error occurs during the request.
            A2AClientJSONError: If the response body cannot be decoded as JSON or validated.
        """
        if not request.id:
            request.id = str(uuid4())

        return SendMessageResponse(
            **await self._send_request(
                request.model_dump(mode='json', exclude_none=True),
                http_kwargs,
            )
        )

    async def send_message_streaming(
        self,
        request: SendStreamingMessageRequest,
        *,
        http_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[SendStreamingMessageResponse]:
        """Sends a streaming message request to the agent and yields responses as they arrive.

        This method uses Server-Sent Events (SSE) to receive a stream of updates from the agent.

        Args:
            request: The `SendStreamingMessageRequest` object containing the message and configuration.
            http_kwargs: Optional dictionary of keyword arguments to pass to the
                underlying httpx.post request. A default `timeout=None` is set but can be overridden.

        Yields:
            `SendStreamingMessageResponse` objects as they are received in the SSE stream.
            These can be Task, Message, TaskStatusUpdateEvent, or TaskArtifactUpdateEvent.

        Raises:
            A2AClientHTTPError: If an HTTP or SSE protocol error occurs during the request.
            A2AClientJSONError: If an SSE event data cannot be decoded as JSON or validated.
        """
        if not request.id:
            request.id = str(uuid4())

        # Default to no timeout for streaming, can be overridden by http_kwargs
        http_kwargs_with_timeout: dict[str, Any] = {
            'timeout': None,
            **(http_kwargs or {}),
        }

        async with aconnect_sse(
            self.httpx_client,
            'POST',
            self.url,
            json=request.model_dump(mode='json', exclude_none=True),
            **http_kwargs_with_timeout,
        ) as event_source:
            try:
                async for sse in event_source.aiter_sse():
                    yield SendStreamingMessageResponse(**json.loads(sse.data))
            except SSEError as e:
                raise A2AClientHTTPError(
                    400,
                    f'Invalid SSE response or protocol error: {e}',
                ) from e
            except json.JSONDecodeError as e:
                raise A2AClientJSONError(str(e)) from e
            except httpx.RequestError as e:
                raise A2AClientHTTPError(
                    503, f'Network communication error: {e}'
                ) from e

    async def _send_request(
        self,
        rpc_request_payload: dict[str, Any],
        http_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Sends a non-streaming JSON-RPC request to the agent.

        Args:
            rpc_request_payload: JSON RPC payload for sending the request.
            http_kwargs: Optional dictionary of keyword arguments to pass to the
                underlying httpx.post request.

        Returns:
            The JSON response payload as a dictionary.

        Raises:
            A2AClientHTTPError: If an HTTP error occurs during the request.
            A2AClientJSONError: If the response body cannot be decoded as JSON.
        """
        try:
            response = await self.httpx_client.post(
                self.url, json=rpc_request_payload, **(http_kwargs or {})
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise A2AClientHTTPError(e.response.status_code, str(e)) from e
        except json.JSONDecodeError as e:
            raise A2AClientJSONError(str(e)) from e
        except httpx.RequestError as e:
            raise A2AClientHTTPError(
                503, f'Network communication error: {e}'
            ) from e

    async def get_task(
        self,
        request: GetTaskRequest,
        *,
        http_kwargs: dict[str, Any] | None = None,
    ) -> GetTaskResponse:
        """Retrieves the current state and history of a specific task.

        Args:
            request: The `GetTaskRequest` object specifying the task ID and history length.
            http_kwargs: Optional dictionary of keyword arguments to pass to the
                underlying httpx.post request.

        Returns:
            A `GetTaskResponse` object containing the Task or an error.

        Raises:
            A2AClientHTTPError: If an HTTP error occurs during the request.
            A2AClientJSONError: If the response body cannot be decoded as JSON or validated.
        """
        if not request.id:
            request.id = str(uuid4())

        return GetTaskResponse(
            **await self._send_request(
                request.model_dump(mode='json', exclude_none=True),
                http_kwargs,
            )
        )

    async def cancel_task(
        self,
        request: CancelTaskRequest,
        *,
        http_kwargs: dict[str, Any] | None = None,
    ) -> CancelTaskResponse:
        """Requests the agent to cancel a specific task.

        Args:
            request: The `CancelTaskRequest` object specifying the task ID.
            http_kwargs: Optional dictionary of keyword arguments to pass to the
                underlying httpx.post request.

        Returns:
            A `CancelTaskResponse` object containing the updated Task with canceled status or an error.

        Raises:
            A2AClientHTTPError: If an HTTP error occurs during the request.
            A2AClientJSONError: If the response body cannot be decoded as JSON or validated.
        """
        if not request.id:
            request.id = str(uuid4())

        return CancelTaskResponse(
            **await self._send_request(
                request.model_dump(mode='json', exclude_none=True),
                http_kwargs,
            )
        )

    async def set_task_callback(
        self,
        request: SetTaskPushNotificationConfigRequest,
        *,
        http_kwargs: dict[str, Any] | None = None,
    ) -> SetTaskPushNotificationConfigResponse:
        """Sets or updates the push notification configuration for a specific task.

        Args:
            request: The `SetTaskPushNotificationConfigRequest` object specifying the task ID and configuration.
            http_kwargs: Optional dictionary of keyword arguments to pass to the
                underlying httpx.post request.

        Returns:
            A `SetTaskPushNotificationConfigResponse` object containing the confirmation or an error.

        Raises:
            A2AClientHTTPError: If an HTTP error occurs during the request.
            A2AClientJSONError: If the response body cannot be decoded as JSON or validated.
        """
        if not request.id:
            request.id = str(uuid4())

        return SetTaskPushNotificationConfigResponse(
            **await self._send_request(
                request.model_dump(mode='json', exclude_none=True),
                http_kwargs,
            )
        )

    async def get_task_callback(
        self,
        request: GetTaskPushNotificationConfigRequest,
        *,
        http_kwargs: dict[str, Any] | None = None,
    ) -> GetTaskPushNotificationConfigResponse:
        """Retrieves the push notification configuration for a specific task.

        Args:
            request: The `GetTaskPushNotificationConfigRequest` object specifying the task ID.
            http_kwargs: Optional dictionary of keyword arguments to pass to the
                underlying httpx.post request.

        Returns:
            A `GetTaskPushNotificationConfigResponse` object containing the configuration or an error.

        Raises:
            A2AClientHTTPError: If an HTTP error occurs during the request.
            A2AClientJSONError: If the response body cannot be decoded as JSON or validated.
        """
        if not request.id:
            request.id = str(uuid4())

        return GetTaskPushNotificationConfigResponse(
            **await self._send_request(
                request.model_dump(mode='json', exclude_none=True),
                http_kwargs,
            )
        )
````

## File: src/a2a/server/agent_execution/__init__.py
````python
"""Components for executing agent logic within the A2A server."""

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.agent_execution.request_context_builder import (
    RequestContextBuilder,
)
from a2a.server.agent_execution.simple_request_context_builder import (
    SimpleRequestContextBuilder,
)


__all__ = [
    'AgentExecutor',
    'RequestContext',
    'RequestContextBuilder',
    'SimpleRequestContextBuilder',
]
````

## File: src/a2a/server/request_handlers/request_handler.py
````python
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from a2a.server.context import ServerCallContext
from a2a.server.events.event_queue import Event
from a2a.types import (
    Message,
    MessageSendParams,
    Task,
    TaskIdParams,
    TaskPushNotificationConfig,
    TaskQueryParams,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError


class RequestHandler(ABC):
    """A2A request handler interface.

    This interface defines the methods that an A2A server implementation must
    provide to handle incoming JSON-RPC requests.
    """

    @abstractmethod
    async def on_get_task(
        self,
        params: TaskQueryParams,
        context: ServerCallContext | None = None,
    ) -> Task | None:
        """Handles the 'tasks/get' method.

        Retrieves the state and history of a specific task.

        Args:
            params: Parameters specifying the task ID and optionally history length.
            context: Context provided by the server.

        Returns:
            The `Task` object if found, otherwise `None`.
        """

    @abstractmethod
    async def on_cancel_task(
        self,
        params: TaskIdParams,
        context: ServerCallContext | None = None,
    ) -> Task | None:
        """Handles the 'tasks/cancel' method.

        Requests the agent to cancel an ongoing task.

        Args:
            params: Parameters specifying the task ID.
            context: Context provided by the server.

        Returns:
            The `Task` object with its status updated to canceled, or `None` if the task was not found.
        """

    @abstractmethod
    async def on_message_send(
        self,
        params: MessageSendParams,
        context: ServerCallContext | None = None,
    ) -> Task | Message:
        """Handles the 'message/send' method (non-streaming).

        Sends a message to the agent to create, continue, or restart a task,
        and waits for the final result (Task or Message).

        Args:
            params: Parameters including the message and configuration.
            context: Context provided by the server.

        Returns:
            The final `Task` object or a final `Message` object.
        """

    @abstractmethod
    async def on_message_send_stream(
        self,
        params: MessageSendParams,
        context: ServerCallContext | None = None,
    ) -> AsyncGenerator[Event]:
        """Handles the 'message/stream' method (streaming).

        Sends a message to the agent and yields stream events as they are
        produced (Task updates, Message chunks, Artifact updates).

        Args:
            params: Parameters including the message and configuration.
            context: Context provided by the server.

        Yields:
            `Event` objects from the agent's execution.

        Raises:
             ServerError(UnsupportedOperationError): By default, if not implemented.
        """
        raise ServerError(error=UnsupportedOperationError())
        yield

    @abstractmethod
    async def on_set_task_push_notification_config(
        self,
        params: TaskPushNotificationConfig,
        context: ServerCallContext | None = None,
    ) -> TaskPushNotificationConfig:
        """Handles the 'tasks/pushNotificationConfig/set' method.

        Sets or updates the push notification configuration for a task.

        Args:
            params: Parameters including the task ID and push notification configuration.
            context: Context provided by the server.

        Returns:
            The provided `TaskPushNotificationConfig` upon success.
        """

    @abstractmethod
    async def on_get_task_push_notification_config(
        self,
        params: TaskIdParams,
        context: ServerCallContext | None = None,
    ) -> TaskPushNotificationConfig:
        """Handles the 'tasks/pushNotificationConfig/get' method.

        Retrieves the current push notification configuration for a task.

        Args:
            params: Parameters including the task ID.
            context: Context provided by the server.

        Returns:
            The `TaskPushNotificationConfig` for the task.
        """

    @abstractmethod
    async def on_resubscribe_to_task(
        self,
        params: TaskIdParams,
        context: ServerCallContext | None = None,
    ) -> AsyncGenerator[Event]:
        """Handles the 'tasks/resubscribe' method.

        Allows a client to re-subscribe to a running streaming task's event stream.

        Args:
            params: Parameters including the task ID.
            context: Context provided by the server.

        Yields:
             `Event` objects from the agent's ongoing execution for the specified task.

        Raises:
             ServerError(UnsupportedOperationError): By default, if not implemented.
        """
        raise ServerError(error=UnsupportedOperationError())
        yield
````

## File: src/a2a/server/tasks/__init__.py
````python
"""Components for managing tasks within the A2A server."""

from a2a.server.tasks.inmemory_push_notifier import InMemoryPushNotifier
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.push_notifier import PushNotifier
from a2a.server.tasks.result_aggregator import ResultAggregator
from a2a.server.tasks.task_manager import TaskManager
from a2a.server.tasks.task_store import TaskStore
from a2a.server.tasks.task_updater import TaskUpdater


__all__ = [
    'InMemoryPushNotifier',
    'InMemoryTaskStore',
    'PushNotifier',
    'ResultAggregator',
    'TaskManager',
    'TaskStore',
    'TaskUpdater',
]
````

## File: src/a2a/server/tasks/task_manager.py
````python
import logging

from a2a.server.events.event_queue import Event
from a2a.server.tasks.task_store import TaskStore
from a2a.types import (
    InvalidParamsError,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import append_artifact_to_task
from a2a.utils.errors import ServerError


logger = logging.getLogger(__name__)


class TaskManager:
    """Helps manage a task's lifecycle during execution of a request.

    Responsible for retrieving, saving, and updating the `Task` object based on
    events received from the agent.
    """

    def __init__(
        self,
        task_id: str | None,
        context_id: str | None,
        task_store: TaskStore,
        initial_message: Message | None,
    ):
        """Initializes the TaskManager.

        Args:
            task_id: The ID of the task, if known from the request.
            context_id: The ID of the context, if known from the request.
            task_store: The `TaskStore` instance for persistence.
            initial_message: The `Message` that initiated the task, if any.
                             Used when creating a new task object.
        """
        self.task_id = task_id
        self.context_id = context_id
        self.task_store = task_store
        self._initial_message = initial_message
        self._current_task: Task | None = None
        logger.debug(
            'TaskManager initialized with task_id: %s, context_id: %s',
            task_id,
            context_id,
        )

    async def get_task(self) -> Task | None:
        """Retrieves the current task object, either from memory or the store.

        If `task_id` is set, it first checks the in-memory `_current_task`,
        then attempts to load it from the `task_store`.

        Returns:
            The `Task` object if found, otherwise `None`.
        """
        if not self.task_id:
            logger.debug('task_id is not set, cannot get task.')
            return None

        if self._current_task:
            return self._current_task

        logger.debug(
            'Attempting to get task from store with id: %s', self.task_id
        )
        self._current_task = await self.task_store.get(self.task_id)
        if self._current_task:
            logger.debug('Task %s retrieved successfully.', self.task_id)
        else:
            logger.debug('Task %s not found.', self.task_id)
        return self._current_task

    async def save_task_event(
        self, event: Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
    ) -> Task | None:
        """Processes a task-related event (Task, Status, Artifact) and saves the updated task state.

        Ensures task and context IDs match or are set from the event.

        Args:
            event: The task-related event (`Task`, `TaskStatusUpdateEvent`, or `TaskArtifactUpdateEvent`).

        Returns:
            The updated `Task` object after processing the event.

        Raises:
            ServerError: If the task ID in the event conflicts with the TaskManager's ID
                         when the TaskManager's ID is already set.
        """
        task_id_from_event = (
            event.id if isinstance(event, Task) else event.taskId
        )
        # If task id is known, make sure it is matched
        if self.task_id and self.task_id != task_id_from_event:
            raise ServerError(
                error=InvalidParamsError(
                    message=f"Task in event doesn't match TaskManager {self.task_id} : {task_id_from_event}"
                )
            )
        if not self.task_id:
            self.task_id = task_id_from_event
        if not self.context_id and self.context_id != event.contextId:
            self.context_id = event.contextId

        logger.debug(
            'Processing save of task event of type %s for task_id: %s',
            type(event).__name__,
            task_id_from_event,
        )
        if isinstance(event, Task):
            await self._save_task(event)
            return event

        task: Task = await self.ensure_task(event)

        if isinstance(event, TaskStatusUpdateEvent):
            logger.debug(
                'Updating task %s status to: %s', task.id, event.status.state
            )
            if task.status.message:
                if not task.history:
                    task.history = [task.status.message]
                else:
                    task.history.append(task.status.message)

            task.status = event.status
        else:
            logger.debug('Appending artifact to task %s', task.id)
            append_artifact_to_task(task, event)

        await self._save_task(task)
        return task

    async def ensure_task(
        self, event: TaskStatusUpdateEvent | TaskArtifactUpdateEvent
    ) -> Task:
        """Ensures a Task object exists in memory, loading from store or creating new if needed.

        Args:
            event: The task-related event triggering the need for a Task object.

        Returns:
            An existing or newly created `Task` object.
        """
        task: Task | None = self._current_task
        if not task and self.task_id:
            logger.debug(
                'Attempting to retrieve existing task with id: %s', self.task_id
            )
            task = await self.task_store.get(self.task_id)

        if not task:
            logger.info(
                'Task not found or task_id not set. Creating new task for event (task_id: %s, context_id: %s).',
                event.taskId,
                event.contextId,
            )
            # streaming agent did not previously stream task object.
            # Create a task object with the available information and persist the event
            task = self._init_task_obj(event.taskId, event.contextId)
            await self._save_task(task)

        return task

    async def process(self, event: Event) -> Event:
        """Processes an event, updates the task state if applicable, stores it, and returns the event.

        If the event is task-related (`Task`, `TaskStatusUpdateEvent`, `TaskArtifactUpdateEvent`),
        the internal task state is updated and persisted.

        Args:
            event: The event object received from the agent.

        Returns:
            The same event object that was processed.
        """
        if isinstance(
            event, Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
        ):
            await self.save_task_event(event)

        return event

    def _init_task_obj(self, task_id: str, context_id: str) -> Task:
        """Initializes a new task object in memory.

        Args:
            task_id: The ID for the new task.
            context_id: The context ID for the new task.

        Returns:
            A new `Task` object with initial status and potentially the initial message in history.
        """
        logger.debug(
            'Initializing new Task object with task_id: %s, context_id: %s',
            task_id,
            context_id,
        )
        history = [self._initial_message] if self._initial_message else []
        return Task(
            id=task_id,
            contextId=context_id,
            status=TaskStatus(state=TaskState.submitted),
            history=history,
        )

    async def _save_task(self, task: Task) -> None:
        """Saves the given task to the task store and updates the in-memory `_current_task`.

        Args:
            task: The `Task` object to save.
        """
        logger.debug('Saving task with id: %s', task.id)
        await self.task_store.save(task)
        self._current_task = task
        if not self.task_id:
            logger.info('New task created with id: %s', task.id)
            self.task_id = task.id
            self.context_id = task.contextId

    def update_with_message(self, message: Message, task: Task) -> Task:
        """Updates a task object in memory by adding a new message to its history.

        If the task has a message in its current status, that message is moved
        to the history first.

        Args:
            message: The new `Message` to add to the history.
            task: The `Task` object to update.

        Returns:
            The updated `Task` object (updated in-place).
        """
        if task.status.message:
            if task.history:
                task.history.append(task.status.message)
            else:
                task.history = [task.status.message]
            task.status.message = None
        if task.history:
            task.history.append(message)
        else:
            task.history = [message]
        self._current_task = task
        return task
````

## File: src/a2a/utils/artifact.py
````python
"""Utility functions for creating A2A Artifact objects."""

import uuid

from typing import Any

from a2a.types import Artifact, DataPart, Part, TextPart


def new_artifact(
    parts: list[Part], name: str, description: str = ''
) -> Artifact:
    """Creates a new Artifact object.

    Args:
        parts: The list of `Part` objects forming the artifact's content.
        name: The human-readable name of the artifact.
        description: An optional description of the artifact.

    Returns:
        A new `Artifact` object with a generated artifactId.
    """
    return Artifact(
        artifactId=str(uuid.uuid4()),
        parts=parts,
        name=name,
        description=description,
    )


def new_text_artifact(
    name: str,
    text: str,
    description: str = '',
) -> Artifact:
    """Creates a new Artifact object containing only a single TextPart.

    Args:
        name: The human-readable name of the artifact.
        text: The text content of the artifact.
        description: An optional description of the artifact.

    Returns:
        A new `Artifact` object with a generated artifactId.
    """
    return new_artifact(
        [Part(root=TextPart(text=text))],
        name,
        description,
    )


def new_data_artifact(
    name: str,
    data: dict[str, Any],
    description: str = '',
) -> Artifact:
    """Creates a new Artifact object containing only a single DataPart.

    Args:
        name: The human-readable name of the artifact.
        data: The structured data content of the artifact.
        description: An optional description of the artifact.

    Returns:
        A new `Artifact` object with a generated artifactId.
    """
    return new_artifact(
        [Part(root=DataPart(data=data))],
        name,
        description,
    )
````

## File: tests/client/test_client.py
````python
import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx_sse import EventSource, ServerSentEvent
from pydantic import ValidationError as PydanticValidationError

from a2a.client import (A2ACardResolver, A2AClient, A2AClientHTTPError,
                        A2AClientJSONError, create_text_message_object)
from a2a.types import (A2ARequest, AgentCapabilities, AgentCard, AgentSkill,
                       CancelTaskRequest, CancelTaskResponse,
                       CancelTaskSuccessResponse, GetTaskRequest,
                       GetTaskResponse, InvalidParamsError,
                       JSONRPCErrorResponse, MessageSendParams, Role,
                       SendMessageRequest, SendMessageResponse,
                       SendMessageSuccessResponse, SendStreamingMessageRequest,
                       SendStreamingMessageResponse, TaskIdParams,
                       TaskNotCancelableError, TaskQueryParams)

AGENT_CARD = AgentCard(
    name='Hello World Agent',
    description='Just a hello world agent',
    url='http://localhost:9999/',
    version='1.0.0',
    defaultInputModes=['text'],
    defaultOutputModes=['text'],
    capabilities=AgentCapabilities(),
    skills=[
        AgentSkill(
            id='hello_world',
            name='Returns hello world',
            description='just returns hello world',
            tags=['hello world'],
            examples=['hi', 'hello world'],
        )
    ],
)

AGENT_CARD_EXTENDED = AGENT_CARD.model_copy(
    update={
        'name': 'Hello World Agent - Extended Edition',
        'skills': AGENT_CARD.skills
        + [
            AgentSkill(
                id='extended_skill',
                name='Super Greet',
                description='A more enthusiastic greeting.',
                tags=['extended'],
                examples=['super hi'],
            )
        ],
        'version': '1.0.1',
    }
)

AGENT_CARD_SUPPORTS_EXTENDED = AGENT_CARD.model_copy(
    update={'supportsAuthenticatedExtendedCard': True}
)
AGENT_CARD_NO_URL_SUPPORTS_EXTENDED = AGENT_CARD_SUPPORTS_EXTENDED.model_copy(
    update={'url': ''}
)

MINIMAL_TASK: dict[str, Any] = {
    'id': 'task-abc',
    'contextId': 'session-xyz',
    'status': {'state': 'working'},
    'kind': 'task',
}

MINIMAL_CANCELLED_TASK: dict[str, Any] = {
    'id': 'task-abc',
    'contextId': 'session-xyz',
    'status': {'state': 'canceled'},
    'kind': 'task',
}


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    return AsyncMock(spec=httpx.AsyncClient)


@pytest.fixture
def mock_agent_card() -> MagicMock:
    return MagicMock(spec=AgentCard, url='http://agent.example.com/api')


async def async_iterable_from_list(
    items: list[ServerSentEvent],
) -> AsyncGenerator[ServerSentEvent]:
    """Helper to create an async iterable from a list."""
    for item in items:
        yield item


class TestA2ACardResolver:
    BASE_URL = 'http://example.com'
    AGENT_CARD_PATH = '/.well-known/agent.json'
    FULL_AGENT_CARD_URL = f'{BASE_URL}{AGENT_CARD_PATH}'
    EXTENDED_AGENT_CARD_PATH = '/agent/authenticatedExtendedCard' # Default path

    @pytest.mark.asyncio
    async def test_init_strips_slashes(self, mock_httpx_client: AsyncMock):
        resolver = A2ACardResolver(
            httpx_client=mock_httpx_client,
            base_url='http://example.com/',
            agent_card_path='/.well-known/agent.json/',
        )
        assert resolver.base_url == 'http://example.com'
        assert (
            resolver.agent_card_path == '.well-known/agent.json/'
        )  # Path is only lstrip'd

    @pytest.mark.asyncio
    async def test_get_agent_card_success_public_only(
        self, mock_httpx_client: AsyncMock
    ):
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = AGENT_CARD.model_dump(mode='json')
        mock_httpx_client.get.return_value = mock_response

        resolver = A2ACardResolver(
            httpx_client=mock_httpx_client,
            base_url=self.BASE_URL,
            agent_card_path=self.AGENT_CARD_PATH,
        )
        agent_card = await resolver.get_agent_card(http_kwargs={'timeout': 10})

        mock_httpx_client.get.assert_called_once_with(
            self.FULL_AGENT_CARD_URL, timeout=10
        )
        mock_response.raise_for_status.assert_called_once()
        assert isinstance(agent_card, AgentCard)
        assert agent_card == AGENT_CARD
        # Ensure only one call was made (for the public card)
        assert mock_httpx_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_get_agent_card_success_with_specified_path_for_extended_card(
        self, mock_httpx_client: AsyncMock):
        extended_card_response = AsyncMock(spec=httpx.Response)
        extended_card_response.status_code = 200
        extended_card_response.json.return_value = AGENT_CARD_EXTENDED.model_dump(
            mode='json'
        )

        # Mock the single call for the extended card
        mock_httpx_client.get.return_value = extended_card_response

        resolver = A2ACardResolver(
            httpx_client=mock_httpx_client,
            base_url=self.BASE_URL,
            agent_card_path=self.AGENT_CARD_PATH,
        )
        
        # Fetch the extended card by providing its relative path and example auth
        auth_kwargs = {"headers": {"Authorization": "Bearer test token"}}
        agent_card_result = await resolver.get_agent_card(
            relative_card_path=self.EXTENDED_AGENT_CARD_PATH,
            http_kwargs=auth_kwargs
        )

        expected_extended_url = f'{self.BASE_URL}/{self.EXTENDED_AGENT_CARD_PATH.lstrip("/")}'
        mock_httpx_client.get.assert_called_once_with(expected_extended_url, **auth_kwargs)
        extended_card_response.raise_for_status.assert_called_once()

        assert isinstance(agent_card_result, AgentCard)
        assert agent_card_result == AGENT_CARD_EXTENDED # Should return the extended card

    @pytest.mark.asyncio
    async def test_get_agent_card_validation_error(
        self, mock_httpx_client: AsyncMock
    ):
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        # Data that will cause a Pydantic ValidationError
        mock_response.json.return_value = {"invalid_field": "value", "name": "Test Agent"} 
        mock_httpx_client.get.return_value = mock_response

        resolver = A2ACardResolver(
            httpx_client=mock_httpx_client, base_url=self.BASE_URL
        )
        # The call that is expected to raise an error should be within pytest.raises
        with pytest.raises(A2AClientJSONError) as exc_info:
            await resolver.get_agent_card() # Fetches from default path
        
        assert f'Failed to validate agent card structure from {self.FULL_AGENT_CARD_URL}' in str(exc_info.value)
        assert 'invalid_field' in str(exc_info.value) # Check if Pydantic error details are present
        assert mock_httpx_client.get.call_count == 1 # Should only be called once

    @pytest.mark.asyncio
    async def test_get_agent_card_http_status_error(
        self, mock_httpx_client: AsyncMock
    ):
        mock_response = MagicMock(
            spec=httpx.Response
        )  # Use MagicMock for response attribute
        mock_response.status_code = 404
        mock_response.text = 'Not Found'

        http_status_error = httpx.HTTPStatusError(
            'Not Found', request=MagicMock(), response=mock_response
        )
        mock_httpx_client.get.side_effect = http_status_error

        resolver = A2ACardResolver(
            httpx_client=mock_httpx_client,
            base_url=self.BASE_URL,
            agent_card_path=self.AGENT_CARD_PATH,
        )

        with pytest.raises(A2AClientHTTPError) as exc_info:
            await resolver.get_agent_card()

        assert exc_info.value.status_code == 404
        assert f'Failed to fetch agent card from {self.FULL_AGENT_CARD_URL}' in str(exc_info.value)
        assert 'Not Found' in str(exc_info.value)
        mock_httpx_client.get.assert_called_once_with(self.FULL_AGENT_CARD_URL)

    @pytest.mark.asyncio
    async def test_get_agent_card_json_decode_error(
        self, mock_httpx_client: AsyncMock
    ):
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        # Define json_error before using it
        json_error = json.JSONDecodeError('Expecting value', 'doc', 0)
        mock_response.json.side_effect = json_error
        mock_httpx_client.get.return_value = mock_response

        resolver = A2ACardResolver(
            httpx_client=mock_httpx_client,
            base_url=self.BASE_URL,
            agent_card_path=self.AGENT_CARD_PATH,
        )

        with pytest.raises(A2AClientJSONError) as exc_info:
            await resolver.get_agent_card()

        # Assertions using exc_info must be after the with block
        assert f'Failed to parse JSON for agent card from {self.FULL_AGENT_CARD_URL}' in str(exc_info.value)
        assert 'Expecting value' in str(exc_info.value)
        mock_httpx_client.get.assert_called_once_with(self.FULL_AGENT_CARD_URL)

    @pytest.mark.asyncio
    async def test_get_agent_card_request_error(
        self, mock_httpx_client: AsyncMock
    ):
        request_error = httpx.RequestError('Network issue', request=MagicMock())
        mock_httpx_client.get.side_effect = request_error

        resolver = A2ACardResolver(
            httpx_client=mock_httpx_client,
            base_url=self.BASE_URL,
            agent_card_path=self.AGENT_CARD_PATH,
        )

        with pytest.raises(A2AClientHTTPError) as exc_info:
            await resolver.get_agent_card()

        assert exc_info.value.status_code == 503
        assert f'Network communication error fetching agent card from {self.FULL_AGENT_CARD_URL}' in str(exc_info.value)
        assert 'Network issue' in str(exc_info.value)
        mock_httpx_client.get.assert_called_once_with(self.FULL_AGENT_CARD_URL)


class TestA2AClient:
    AGENT_URL = 'http://agent.example.com/api'

    def test_init_with_agent_card(
        self, mock_httpx_client: AsyncMock, mock_agent_card: MagicMock
    ):
        client = A2AClient(
            httpx_client=mock_httpx_client, agent_card=mock_agent_card
        )
        assert client.url == mock_agent_card.url
        assert client.httpx_client == mock_httpx_client

    def test_init_with_url(self, mock_httpx_client: AsyncMock):
        client = A2AClient(httpx_client=mock_httpx_client, url=self.AGENT_URL)
        assert client.url == self.AGENT_URL
        assert client.httpx_client == mock_httpx_client

    def test_init_with_agent_card_and_url_prioritizes_agent_card(
        self, mock_httpx_client: AsyncMock, mock_agent_card: MagicMock
    ):
        client = A2AClient(
            httpx_client=mock_httpx_client,
            agent_card=mock_agent_card,
            url='http://otherurl.com',
        )
        assert (
            client.url == mock_agent_card.url
        )  # Agent card URL should be used

    def test_init_raises_value_error_if_no_card_or_url(
        self, mock_httpx_client: AsyncMock
    ):
        with pytest.raises(ValueError) as exc_info:
            A2AClient(httpx_client=mock_httpx_client)
        assert 'Must provide either agent_card or url' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_client_from_agent_card_url_success(
        self, mock_httpx_client: AsyncMock, mock_agent_card: MagicMock
    ):
        base_url = 'http://example.com'
        agent_card_path = '/.well-known/custom-agent.json'
        resolver_kwargs = {'timeout': 30}

        mock_resolver_instance = AsyncMock(spec=A2ACardResolver)
        mock_resolver_instance.get_agent_card.return_value = mock_agent_card

        with patch(
            'a2a.client.client.A2ACardResolver',
            return_value=mock_resolver_instance,
        ) as mock_resolver_class:
            client = await A2AClient.get_client_from_agent_card_url(
                httpx_client=mock_httpx_client,
                base_url=base_url,
                agent_card_path=agent_card_path,
                http_kwargs=resolver_kwargs,
            )

            mock_resolver_class.assert_called_once_with(
                mock_httpx_client,
                base_url=base_url,
                agent_card_path=agent_card_path,
            )
            mock_resolver_instance.get_agent_card.assert_called_once_with(
                http_kwargs=resolver_kwargs,
                # relative_card_path=None is implied by not passing it
            )
            assert isinstance(client, A2AClient)
            assert client.url == mock_agent_card.url
            assert client.httpx_client == mock_httpx_client

    @pytest.mark.asyncio
    async def test_get_client_from_agent_card_url_resolver_error(
        self, mock_httpx_client: AsyncMock
    ):
        error_to_raise = A2AClientHTTPError(404, 'Agent card not found')
        with patch(
            'a2a.client.client.A2ACardResolver.get_agent_card',
            new_callable=AsyncMock,
            side_effect=error_to_raise,
        ):
            with pytest.raises(A2AClientHTTPError) as exc_info:
                await A2AClient.get_client_from_agent_card_url(
                    httpx_client=mock_httpx_client,
                    base_url='http://example.com',
                )
            assert exc_info.value == error_to_raise

    @pytest.mark.asyncio
    async def test_send_message_success_use_request(
        self, mock_httpx_client: AsyncMock, mock_agent_card: MagicMock
    ):
        client = A2AClient(
            httpx_client=mock_httpx_client, agent_card=mock_agent_card
        )

        params = MessageSendParams(
            message=create_text_message_object(content='Hello')
        )

        request = SendMessageRequest(id=123, params=params)

        success_response = create_text_message_object(
            role=Role.agent, content='Hi there!'
        ).model_dump(exclude_none=True)

        rpc_response: dict[str, Any] = {
            'id': 123,
            'jsonrpc': '2.0',
            'result': success_response,
        }

        with patch.object(
            client, '_send_request', new_callable=AsyncMock
        ) as mock_send_req:
            mock_send_req.return_value = rpc_response
            response = await client.send_message(
                request=request, http_kwargs={'timeout': 10}
            )

            assert mock_send_req.call_count == 1
            called_args, called_kwargs = mock_send_req.call_args
            assert not called_kwargs  # no kwargs to _send_request
            assert len(called_args) == 2
            json_rpc_request: dict[str, Any] = called_args[0]
            assert isinstance(json_rpc_request['id'], int)
            http_kwargs: dict[str, Any] = called_args[1]
            assert http_kwargs['timeout'] == 10

            a2a_request_arg = A2ARequest.model_validate(json_rpc_request)
            assert isinstance(a2a_request_arg.root, SendMessageRequest)
            assert isinstance(a2a_request_arg.root.params, MessageSendParams)

            assert a2a_request_arg.root.params.model_dump(
                exclude_none=True
            ) == params.model_dump(exclude_none=True)

            assert isinstance(response, SendMessageResponse)
            assert isinstance(response.root, SendMessageSuccessResponse)
            assert (
                response.root.result.model_dump(exclude_none=True)
                == success_response
            )

    @pytest.mark.asyncio
    async def test_send_message_error_response(
        self, mock_httpx_client: AsyncMock, mock_agent_card: MagicMock
    ):
        client = A2AClient(
            httpx_client=mock_httpx_client, agent_card=mock_agent_card
        )

        params = MessageSendParams(
            message=create_text_message_object(content='Hello')
        )

        request = SendMessageRequest(id=123, params=params)

        error_response = InvalidParamsError()

        rpc_response: dict[str, Any] = {
            'id': 123,
            'jsonrpc': '2.0',
            'error': error_response.model_dump(exclude_none=True),
        }

        with patch.object(
            client, '_send_request', new_callable=AsyncMock
        ) as mock_send_req:
            mock_send_req.return_value = rpc_response
            response = await client.send_message(request=request)

            assert isinstance(response, SendMessageResponse)
            assert isinstance(response.root, JSONRPCErrorResponse)
            assert response.root.error.model_dump(
                exclude_none=True
            ) == InvalidParamsError().model_dump(exclude_none=True)

    @pytest.mark.asyncio
    @patch('a2a.client.client.aconnect_sse')
    async def test_send_message_streaming_success_request(
        self,
        mock_aconnect_sse: AsyncMock,
        mock_httpx_client: AsyncMock,
        mock_agent_card: MagicMock,
    ):
        client = A2AClient(
            httpx_client=mock_httpx_client, agent_card=mock_agent_card
        )
        params = MessageSendParams(
            message=create_text_message_object(content='Hello stream')
        )

        request = SendStreamingMessageRequest(id=123, params=params)

        mock_stream_response_1_dict: dict[str, Any] = {
            'id': 'stream_id_123',
            'jsonrpc': '2.0',
            'result': create_text_message_object(
                content='First part ', role=Role.agent
            ).model_dump(mode='json', exclude_none=True),
        }
        mock_stream_response_2_dict: dict[str, Any] = {
            'id': 'stream_id_123',
            'jsonrpc': '2.0',
            'result': create_text_message_object(
                content='second part ', role=Role.agent
            ).model_dump(mode='json', exclude_none=True),
        }

        sse_event_1 = ServerSentEvent(
            data=json.dumps(mock_stream_response_1_dict)
        )
        sse_event_2 = ServerSentEvent(
            data=json.dumps(mock_stream_response_2_dict)
        )

        mock_event_source = AsyncMock(spec=EventSource)
        with patch.object(mock_event_source, 'aiter_sse') as mock_aiter_sse:
            mock_aiter_sse.return_value = async_iterable_from_list(
                [sse_event_1, sse_event_2]
            )
            mock_aconnect_sse.return_value.__aenter__.return_value = (
                mock_event_source
            )

            results: list[Any] = []
            async for response in client.send_message_streaming(
                request=request
            ):
                results.append(response)

            assert len(results) == 2
            assert isinstance(results[0], SendStreamingMessageResponse)
            # Assuming SendStreamingMessageResponse is a RootModel like SendMessageResponse
            assert results[0].root.id == 'stream_id_123'
            assert (
                results[0].root.result.model_dump(  # type: ignore
                    mode='json', exclude_none=True
                )
                == mock_stream_response_1_dict['result']
            )

            assert isinstance(results[1], SendStreamingMessageResponse)
            assert results[1].root.id == 'stream_id_123'
            assert (
                results[1].root.result.model_dump(  # type: ignore
                    mode='json', exclude_none=True
                )
                == mock_stream_response_2_dict['result']
            )

            mock_aconnect_sse.assert_called_once()
            call_args, call_kwargs = mock_aconnect_sse.call_args
            assert call_args[0] == mock_httpx_client
            assert call_args[1] == 'POST'
            assert call_args[2] == mock_agent_card.url

            sent_json_payload = call_kwargs['json']
            assert sent_json_payload['method'] == 'message/stream'
            assert sent_json_payload['params'] == params.model_dump(
                mode='json', exclude_none=True
            )
            assert (
                call_kwargs['timeout'] is None
            )  # Default timeout for streaming

    @pytest.mark.asyncio
    async def test_get_task_success_use_request(
        self, mock_httpx_client: AsyncMock, mock_agent_card: MagicMock
    ):
        client = A2AClient(
            httpx_client=mock_httpx_client, agent_card=mock_agent_card
        )
        task_id_val = 'task_for_req_obj'
        params_model = TaskQueryParams(id=task_id_val)
        request_obj_id = 789
        request = GetTaskRequest(id=request_obj_id, params=params_model)

        rpc_response_payload: dict[str, Any] = {
            'id': request_obj_id,
            'jsonrpc': '2.0',
            'result': MINIMAL_TASK,
        }

        with patch.object(
            client, '_send_request', new_callable=AsyncMock
        ) as mock_send_req:
            mock_send_req.return_value = rpc_response_payload
            response = await client.get_task(
                request=request, http_kwargs={'timeout': 20}
            )

            assert mock_send_req.call_count == 1
            called_args, called_kwargs = mock_send_req.call_args
            assert len(called_args) == 2
            json_rpc_request_sent: dict[str, Any] = called_args[0]
            assert not called_kwargs  # no extra kwargs to _send_request
            http_kwargs: dict[str, Any] = called_args[1]
            assert http_kwargs['timeout'] == 20

            assert json_rpc_request_sent['method'] == 'tasks/get'
            assert json_rpc_request_sent['id'] == request_obj_id
            assert json_rpc_request_sent['params'] == params_model.model_dump(
                mode='json', exclude_none=True
            )

            assert isinstance(response, GetTaskResponse)
            assert hasattr(response.root, 'result')
            assert (
                response.root.result.model_dump(mode='json', exclude_none=True)  # type: ignore
                == MINIMAL_TASK
            )
            assert response.root.id == request_obj_id

    @pytest.mark.asyncio
    async def test_get_task_error_response(
        self, mock_httpx_client: AsyncMock, mock_agent_card: MagicMock
    ):
        client = A2AClient(
            httpx_client=mock_httpx_client, agent_card=mock_agent_card
        )
        params_model = TaskQueryParams(id='task_error_case')
        request = GetTaskRequest(id='err_req_id', params=params_model)
        error_details = InvalidParamsError()

        rpc_response_payload: dict[str, Any] = {
            'id': 'err_req_id',
            'jsonrpc': '2.0',
            'error': error_details.model_dump(mode='json', exclude_none=True),
        }

        with patch.object(
            client, '_send_request', new_callable=AsyncMock
        ) as mock_send_req:
            mock_send_req.return_value = rpc_response_payload
            response = await client.get_task(request=request)

            assert isinstance(response, GetTaskResponse)
            assert isinstance(response.root, JSONRPCErrorResponse)
            assert response.root.error.model_dump(
                mode='json', exclude_none=True
            ) == error_details.model_dump(exclude_none=True)
            assert response.root.id == 'err_req_id'

    @pytest.mark.asyncio
    async def test_cancel_task_success_use_request(
        self, mock_httpx_client: AsyncMock, mock_agent_card: MagicMock
    ):
        client = A2AClient(
            httpx_client=mock_httpx_client, agent_card=mock_agent_card
        )
        task_id_val = MINIMAL_CANCELLED_TASK['id']
        params_model = TaskIdParams(id=task_id_val)
        request_obj_id = 'cancel_req_obj_id_001'
        request = CancelTaskRequest(id=request_obj_id, params=params_model)

        rpc_response_payload: dict[str, Any] = {
            'id': request_obj_id,
            'jsonrpc': '2.0',
            'result': MINIMAL_CANCELLED_TASK,
        }

        with patch.object(
            client, '_send_request', new_callable=AsyncMock
        ) as mock_send_req:
            mock_send_req.return_value = rpc_response_payload
            response = await client.cancel_task(
                request=request, http_kwargs={'timeout': 15}
            )

            assert mock_send_req.call_count == 1
            called_args, called_kwargs = mock_send_req.call_args
            assert not called_kwargs  # no extra kwargs to _send_request
            assert len(called_args) == 2
            json_rpc_request_sent: dict[str, Any] = called_args[0]
            http_kwargs: dict[str, Any] = called_args[1]
            assert http_kwargs['timeout'] == 15

            assert json_rpc_request_sent['method'] == 'tasks/cancel'
            assert json_rpc_request_sent['id'] == request_obj_id
            assert json_rpc_request_sent['params'] == params_model.model_dump(
                mode='json', exclude_none=True
            )

            assert isinstance(response, CancelTaskResponse)
            assert isinstance(response.root, CancelTaskSuccessResponse)
            assert (
                response.root.result.model_dump(mode='json', exclude_none=True)  # type: ignore
                == MINIMAL_CANCELLED_TASK
            )
            assert response.root.id == request_obj_id

    @pytest.mark.asyncio
    async def test_cancel_task_error_response(
        self, mock_httpx_client: AsyncMock, mock_agent_card: MagicMock
    ):
        client = A2AClient(
            httpx_client=mock_httpx_client, agent_card=mock_agent_card
        )
        params_model = TaskIdParams(id='task_cancel_error_case')
        request = CancelTaskRequest(id='err_cancel_req', params=params_model)
        error_details = TaskNotCancelableError()

        rpc_response_payload: dict[str, Any] = {
            'id': 'err_cancel_req',
            'jsonrpc': '2.0',
            'error': error_details.model_dump(mode='json', exclude_none=True),
        }

        with patch.object(
            client, '_send_request', new_callable=AsyncMock
        ) as mock_send_req:
            mock_send_req.return_value = rpc_response_payload
            response = await client.cancel_task(request=request)

            assert isinstance(response, CancelTaskResponse)
            assert isinstance(response.root, JSONRPCErrorResponse)
            assert response.root.error.model_dump(
                mode='json', exclude_none=True
            ) == error_details.model_dump(exclude_none=True)
            assert response.root.id == 'err_cancel_req'
````

## File: CODE_OF_CONDUCT.md
````markdown
# Code of Conduct

## Our Pledge

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to make participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of
experience, education, socio-economic status, nationality, personal appearance,
race, religion, or sexual identity and orientation.

## Our Standards

Examples of behavior that contributes to creating a positive environment
include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

- The use of sexualized language or imagery and unwelcome sexual attention or
  advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information, such as a physical or electronic
  address, without explicit permission
- Other conduct which could reasonably be considered inappropriate in a
  professional setting

## Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable
behavior and are expected to take appropriate and fair corrective action in
response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or reject
comments, commits, code, wiki edits, issues, and other contributions that are
not aligned to this Code of Conduct, or to ban temporarily or permanently any
contributor for other behaviors that they deem inappropriate, threatening,
offensive, or harmful.

## Scope

This Code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community. Examples of
representing a project or community include using an official project email
address, posting via an official social media account, or acting as an appointed
representative at an online or offline event. Representation of a project may be
further defined and clarified by project maintainers.

This Code of Conduct also applies outside the project spaces when the Project
Steward has a reasonable belief that an individual's behavior may have a
negative impact on the project or its community.

## Conflict Resolution

We do not believe that all conflict is bad; healthy debate and disagreement
often yield positive results. However, it is never okay to be disrespectful or
to engage in behavior that violates the project's code of conduct.

If you see someone violating the code of conduct, you are encouraged to address
the behavior directly with those involved. Many issues can be resolved quickly
and easily, and this gives people more control over the outcome of their
dispute. If you are unable to resolve the matter for any reason, or if the
behavior is threatening or harassing, report it. We are dedicated to providing
an environment where participants feel welcome and safe.

Reports should be directed to _[PROJECT STEWARD NAME(s) AND EMAIL(s)]_, the
Project Steward(s) for _[PROJECT NAME]_. It is the Project Steward's duty to
receive and address reported violations of the code of conduct. They will then
work with a committee consisting of representatives from the Open Source
Programs Office and the Google Open Source Strategy team. If for any reason you
are uncomfortable reaching out to the Project Steward, please email
opensource@google.com.

We will investigate every complaint, but you may not receive a direct response.
We will use our discretion in determining when and how to follow up on reported
incidents, which may range from not taking action to permanent expulsion from
the project and project-sponsored spaces. We will notify the accused of the
report and provide them an opportunity to discuss it before any action is taken.
The identity of the reporter will be omitted from the details of the report
supplied to the accused. In potentially harmful situations, such as ongoing
harassment or threats to anyone's safety, we may take action without notice.

## Attribution

This Code of Conduct is adapted from the Contributor Covenant, version 1.4,
available at
https://www.contributor-covenant.org/version/1/4/code-of-conduct.html

Note: A version of this file is also available in the
[New Project repository](https://github.com/google/new-project/blob/master/docs/code-of-conduct.md).
````

## File: noxfile.py
````python
# pylint: skip-file
# type: ignore
# -*- coding: utf-8 -*-
#
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import subprocess

import nox


DEFAULT_PYTHON_VERSION = '3.10'

CURRENT_DIRECTORY = pathlib.Path(__file__).parent.absolute()

nox.options.sessions = [
    'format',
]

# Error if a python version is missing
nox.options.error_on_missing_interpreters = True


@nox.session(python=DEFAULT_PYTHON_VERSION)
def format(session):
    """Format Python code using autoflake, pyupgrade, and ruff."""
    # Sort Spelling Allowlist
    spelling_allow_file = '.github/actions/spelling/allow.txt'

    with open(spelling_allow_file, encoding='utf-8') as file:
        unique_words = sorted(set(file))

    with open(spelling_allow_file, 'w', encoding='utf-8') as file:
        file.writelines(unique_words)

    format_all = False

    if format_all:
        lint_paths_py = ['.']
    else:
        target_branch = 'origin/main'

        unstaged_files = subprocess.run(
            [
                'git',
                'diff',
                '--name-only',
                '--diff-filter=ACMRTUXB',
                target_branch,
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout.splitlines()

        staged_files = subprocess.run(
            [
                'git',
                'diff',
                '--cached',
                '--name-only',
                '--diff-filter=ACMRTUXB',
                target_branch,
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout.splitlines()

        committed_files = subprocess.run(
            [
                'git',
                'diff',
                'HEAD',
                target_branch,
                '--name-only',
                '--diff-filter=ACMRTUXB',
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout.splitlines()

        changed_files = sorted(
            {
                file
                for file in (unstaged_files + staged_files + committed_files)
                if os.path.isfile(file)
            }
        )

        lint_paths_py = [f for f in changed_files if f.endswith('.py')]

        if not lint_paths_py:
            session.log('No changed Python files to lint.')
            return

    session.install(
        'types-requests',
        'pyupgrade',
        'autoflake',
        'ruff',
        'no_implicit_optional',
    )

    if lint_paths_py:
        session.run(
            'no_implicit_optional',
            '--use-union-or',
            *lint_paths_py,
        )
        if not format_all:
            session.run(
                'pyupgrade',
                '--exit-zero-even-if-changed',
                '--py310-plus',
                *lint_paths_py,
            )
        session.run(
            'autoflake',
            '-i',
            '-r',
            '--remove-all-unused-imports',
            *lint_paths_py,
        )
        session.run(
            'ruff',
            'check',
            '--fix-only',
            *lint_paths_py,
        )
        session.run(
            'ruff',
            'format',
            *lint_paths_py,
        )
````

## File: src/a2a/server/events/event_queue.py
````python
import asyncio
import logging
import sys

from a2a.types import (
    A2AError,
    JSONRPCError,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from a2a.utils.telemetry import SpanKind, trace_class


logger = logging.getLogger(__name__)


Event = (
    Message
    | Task
    | TaskStatusUpdateEvent
    | TaskArtifactUpdateEvent
    | A2AError
    | JSONRPCError
)
"""Type alias for events that can be enqueued."""


@trace_class(kind=SpanKind.SERVER)
class EventQueue:
    """Event queue for A2A responses from agent.

    Acts as a buffer between the agent's asynchronous execution and the
    server's response handling (e.g., streaming via SSE). Supports tapping
    to create child queues that receive the same events.
    """

    def __init__(self) -> None:
        """Initializes the EventQueue."""
        self.queue: asyncio.Queue[Event] = asyncio.Queue()
        self._children: list[EventQueue] = []
        self._is_closed = False
        self._lock = asyncio.Lock()
        logger.debug('EventQueue initialized.')

    def enqueue_event(self, event: Event):
        """Enqueues an event to this queue and all its children.

        Args:
            event: The event object to enqueue.
        """
        if self._is_closed:
            logger.warning('Queue is closed. Event will not be enqueued.')
            return
        logger.debug(f'Enqueuing event of type: {type(event)}')
        self.queue.put_nowait(event)
        for child in self._children:
            child.enqueue_event(event)

    async def dequeue_event(self, no_wait: bool = False) -> Event:
        """Dequeues an event from the queue.

        This implementation expects that dequeue to raise an exception when
        the queue has been closed. In python 3.13+ this is naturally provided
        by the QueueShutDown exception generated when the queue has closed and
        the user is awaiting the queue.get method. Python<=3.12 this needs to
        manage this lifecycle itself. The current implementation can lead to
        blocking if the dequeue_event is called before the EventQueue has been
        closed but when there are no events on the queue. Two ways to avoid this
        are to call this with no_wait = True which won't block, but is the
        callers responsibility to retry as appropriate. Alternatively, one can
        use a async Task management solution to cancel the get task if the queue
        has closed or some other condition is met. The implementation of the
        EventConsumer uses an async.wait with a timeout to abort the
        dequeue_event call and retry, when it will return with a closed error.

        Args:
            no_wait: If True, retrieve an event immediately or raise `asyncio.QueueEmpty`.
                     If False (default), wait until an event is available.

        Returns:
            The next event from the queue.

        Raises:
            asyncio.QueueEmpty: If `no_wait` is True and the queue is empty.
            asyncio.QueueShutDown: If the queue has been closed and is empty.
        """
        async with self._lock:
            if self._is_closed and self.queue.empty():
                logger.warning('Queue is closed. Event will not be dequeued.')
                raise asyncio.QueueEmpty('Queue is closed.')

        if no_wait:
            logger.debug('Attempting to dequeue event (no_wait=True).')
            event = self.queue.get_nowait()
            logger.debug(
                f'Dequeued event (no_wait=True) of type: {type(event)}'
            )
            return event

        logger.debug('Attempting to dequeue event (waiting).')
        event = await self.queue.get()
        logger.debug(f'Dequeued event (waited) of type: {type(event)}')
        return event

    def task_done(self) -> None:
        """Signals that a formerly enqueued task is complete.

        Used in conjunction with `dequeue_event` to track processed items.
        """
        logger.debug('Marking task as done in EventQueue.')
        self.queue.task_done()

    def tap(self) -> 'EventQueue':
        """Taps the event queue to create a new child queue that receives all future events.

        Returns:
            A new `EventQueue` instance that will receive all events enqueued
            to this parent queue from this point forward.
        """
        logger.debug('Tapping EventQueue to create a child queue.')
        queue = EventQueue()
        self._children.append(queue)
        return queue

    async def close(self):
        """Closes the queue for future push events.

        Once closed, `dequeue_event` will eventually raise `asyncio.QueueShutDown`
        when the queue is empty. Also closes all child queues.
        """
        logger.debug('Closing EventQueue.')
        async with self._lock:
            # If already closed, just return.
            if self._is_closed:
                return
            self._is_closed = True
        # If using python 3.13 or higher, use the shutdown method
        if sys.version_info >= (3, 13):
            self.queue.shutdown()
            for child in self._children:
                child.close()
        # Otherwise, join the queue
        else:
            tasks = [asyncio.create_task(self.queue.join())]
            for child in self._children:
                tasks.append(asyncio.create_task(child.close()))
            await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    def is_closed(self) -> bool:
        """Checks if the queue is closed."""
        return self._is_closed
````

## File: src/a2a/server/request_handlers/jsonrpc_handler.py
````python
import logging

from collections.abc import AsyncIterable

from a2a.server.context import ServerCallContext
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.request_handlers.response_helpers import prepare_response_object
from a2a.types import (
    AgentCard,
    CancelTaskRequest,
    CancelTaskResponse,
    CancelTaskSuccessResponse,
    GetTaskPushNotificationConfigRequest,
    GetTaskPushNotificationConfigResponse,
    GetTaskPushNotificationConfigSuccessResponse,
    GetTaskRequest,
    GetTaskResponse,
    GetTaskSuccessResponse,
    InternalError,
    JSONRPCErrorResponse,
    Message,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    SendStreamingMessageRequest,
    SendStreamingMessageResponse,
    SendStreamingMessageSuccessResponse,
    SetTaskPushNotificationConfigRequest,
    SetTaskPushNotificationConfigResponse,
    SetTaskPushNotificationConfigSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskNotFoundError,
    TaskPushNotificationConfig,
    TaskResubscriptionRequest,
    TaskStatusUpdateEvent,
)
from a2a.utils.errors import ServerError
from a2a.utils.helpers import validate
from a2a.utils.telemetry import SpanKind, trace_class


logger = logging.getLogger(__name__)


@trace_class(kind=SpanKind.SERVER)
class JSONRPCHandler:
    """Maps incoming JSON-RPC requests to the appropriate request handler method and formats responses."""

    def __init__(
        self,
        agent_card: AgentCard,
        request_handler: RequestHandler,
    ):
        """Initializes the JSONRPCHandler.

        Args:
            agent_card: The AgentCard describing the agent's capabilities.
            request_handler: The underlying `RequestHandler` instance to delegate requests to.
        """
        self.agent_card = agent_card
        self.request_handler = request_handler

    async def on_message_send(
        self,
        request: SendMessageRequest,
        context: ServerCallContext | None = None,
    ) -> SendMessageResponse:
        """Handles the 'message/send' JSON-RPC method.

        Args:
            request: The incoming `SendMessageRequest` object.
            context: Context provided by the server.

        Returns:
            A `SendMessageResponse` object containing the result (Task or Message)
            or a JSON-RPC error response if a `ServerError` is raised by the handler.
        """
        # TODO: Wrap in error handler to return error states
        try:
            task_or_message = await self.request_handler.on_message_send(
                request.params, context
            )
            return prepare_response_object(
                request.id,
                task_or_message,
                (Task, Message),
                SendMessageSuccessResponse,
                SendMessageResponse,
            )
        except ServerError as e:
            return SendMessageResponse(
                root=JSONRPCErrorResponse(
                    id=request.id, error=e.error if e.error else InternalError()
                )
            )

    @validate(
        lambda self: self.agent_card.capabilities.streaming,
        'Streaming is not supported by the agent',
    )
    async def on_message_send_stream(
        self,
        request: SendStreamingMessageRequest,
        context: ServerCallContext | None = None,
    ) -> AsyncIterable[SendStreamingMessageResponse]:
        """Handles the 'message/stream' JSON-RPC method.

        Yields response objects as they are produced by the underlying handler's stream.

        Args:
            request: The incoming `SendStreamingMessageRequest` object.
            context: Context provided by the server.

        Yields:
            `SendStreamingMessageResponse` objects containing streaming events
            (Task, Message, TaskStatusUpdateEvent, TaskArtifactUpdateEvent)
            or JSON-RPC error responses if a `ServerError` is raised.
        """
        try:
            async for event in self.request_handler.on_message_send_stream(
                request.params, context
            ):
                yield prepare_response_object(
                    request.id,
                    event,
                    (
                        Task,
                        Message,
                        TaskArtifactUpdateEvent,
                        TaskStatusUpdateEvent,
                    ),
                    SendStreamingMessageSuccessResponse,
                    SendStreamingMessageResponse,
                )
        except ServerError as e:
            yield SendStreamingMessageResponse(
                root=JSONRPCErrorResponse(
                    id=request.id, error=e.error if e.error else InternalError()
                )
            )

    async def on_cancel_task(
        self,
        request: CancelTaskRequest,
        context: ServerCallContext | None = None,
    ) -> CancelTaskResponse:
        """Handles the 'tasks/cancel' JSON-RPC method.

        Args:
            request: The incoming `CancelTaskRequest` object.
            context: Context provided by the server.

        Returns:
            A `CancelTaskResponse` object containing the updated Task or a JSON-RPC error.
        """
        try:
            task = await self.request_handler.on_cancel_task(
                request.params, context
            )
            if task:
                return prepare_response_object(
                    request.id,
                    task,
                    (Task,),
                    CancelTaskSuccessResponse,
                    CancelTaskResponse,
                )
            raise ServerError(error=TaskNotFoundError())
        except ServerError as e:
            return CancelTaskResponse(
                root=JSONRPCErrorResponse(
                    id=request.id, error=e.error if e.error else InternalError()
                )
            )

    async def on_resubscribe_to_task(
        self,
        request: TaskResubscriptionRequest,
        context: ServerCallContext | None = None,
    ) -> AsyncIterable[SendStreamingMessageResponse]:
        """Handles the 'tasks/resubscribe' JSON-RPC method.

        Yields response objects as they are produced by the underlying handler's stream.

        Args:
            request: The incoming `TaskResubscriptionRequest` object.
            context: Context provided by the server.

        Yields:
            `SendStreamingMessageResponse` objects containing streaming events
            or JSON-RPC error responses if a `ServerError` is raised.
        """
        try:
            async for event in self.request_handler.on_resubscribe_to_task(
                request.params, context
            ):
                yield prepare_response_object(
                    request.id,
                    event,
                    (
                        Task,
                        Message,
                        TaskArtifactUpdateEvent,
                        TaskStatusUpdateEvent,
                    ),
                    SendStreamingMessageSuccessResponse,
                    SendStreamingMessageResponse,
                )
        except ServerError as e:
            yield SendStreamingMessageResponse(
                root=JSONRPCErrorResponse(
                    id=request.id, error=e.error if e.error else InternalError()
                )
            )

    async def get_push_notification(
        self,
        request: GetTaskPushNotificationConfigRequest,
        context: ServerCallContext | None = None,
    ) -> GetTaskPushNotificationConfigResponse:
        """Handles the 'tasks/pushNotificationConfig/get' JSON-RPC method.

        Args:
            request: The incoming `GetTaskPushNotificationConfigRequest` object.
            context: Context provided by the server.

        Returns:
            A `GetTaskPushNotificationConfigResponse` object containing the config or a JSON-RPC error.
        """
        try:
            config = (
                await self.request_handler.on_get_task_push_notification_config(
                    request.params, context
                )
            )
            return prepare_response_object(
                request.id,
                config,
                (TaskPushNotificationConfig,),
                GetTaskPushNotificationConfigSuccessResponse,
                GetTaskPushNotificationConfigResponse,
            )
        except ServerError as e:
            return GetTaskPushNotificationConfigResponse(
                root=JSONRPCErrorResponse(
                    id=request.id, error=e.error if e.error else InternalError()
                )
            )

    @validate(
        lambda self: self.agent_card.capabilities.pushNotifications,
        'Push notifications are not supported by the agent',
    )
    async def set_push_notification(
        self,
        request: SetTaskPushNotificationConfigRequest,
        context: ServerCallContext | None = None,
    ) -> SetTaskPushNotificationConfigResponse:
        """Handles the 'tasks/pushNotificationConfig/set' JSON-RPC method.

        Requires the agent to support push notifications.

        Args:
            request: The incoming `SetTaskPushNotificationConfigRequest` object.
            context: Context provided by the server.

        Returns:
            A `SetTaskPushNotificationConfigResponse` object containing the config or a JSON-RPC error.

        Raises:
            ServerError: If push notifications are not supported by the agent
                (due to the `@validate` decorator).
        """
        try:
            config = (
                await self.request_handler.on_set_task_push_notification_config(
                    request.params, context
                )
            )
            return prepare_response_object(
                request.id,
                config,
                (TaskPushNotificationConfig,),
                SetTaskPushNotificationConfigSuccessResponse,
                SetTaskPushNotificationConfigResponse,
            )
        except ServerError as e:
            return SetTaskPushNotificationConfigResponse(
                root=JSONRPCErrorResponse(
                    id=request.id, error=e.error if e.error else InternalError()
                )
            )

    async def on_get_task(
        self,
        request: GetTaskRequest,
        context: ServerCallContext | None = None,
    ) -> GetTaskResponse:
        """Handles the 'tasks/get' JSON-RPC method.

        Args:
            request: The incoming `GetTaskRequest` object.
            context: Context provided by the server.

        Returns:
            A `GetTaskResponse` object containing the Task or a JSON-RPC error.
        """
        try:
            task = await self.request_handler.on_get_task(
                request.params, context
            )
            if task:
                return prepare_response_object(
                    request.id,
                    task,
                    (Task,),
                    GetTaskSuccessResponse,
                    GetTaskResponse,
                )
            raise ServerError(error=TaskNotFoundError())
        except ServerError as e:
            return GetTaskResponse(
                root=JSONRPCErrorResponse(
                    id=request.id, error=e.error if e.error else InternalError()
                )
            )
````

## File: src/a2a/utils/message.py
````python
"""Utility functions for creating and handling A2A Message objects."""

import uuid

from a2a.types import (
    Message,
    Part,
    Role,
    TextPart,
)


def new_agent_text_message(
    text: str,
    context_id: str | None = None,
    task_id: str | None = None,
) -> Message:
    """Creates a new agent message containing a single TextPart.

    Args:
        text: The text content of the message.
        context_id: The context ID for the message.
        task_id: The task ID for the message.
        final: Optional boolean indicating if this is the final message.
        metadata: Optional metadata for the message.

    Returns:
        A new `Message` object with role 'agent'.
    """
    return Message(
        role=Role.agent,
        parts=[Part(root=TextPart(text=text))],
        messageId=str(uuid.uuid4()),
        taskId=task_id,
        contextId=context_id,
    )


def new_agent_parts_message(
    parts: list[Part],
    context_id: str | None = None,
    task_id: str | None = None,
):
    """Creates a new agent message containing a list of Parts.

    Args:
        parts: The list of `Part` objects for the message content.
        context_id: The context ID for the message.
        task_id: The task ID for the message.
        final: Optional boolean indicating if this is the final message.
        metadata: Optional metadata for the message.

    Returns:
        A new `Message` object with role 'agent'.
    """
    return Message(
        role=Role.agent,
        parts=parts,
        messageId=str(uuid.uuid4()),
        taskId=task_id,
        contextId=context_id,
    )


def get_text_parts(parts: list[Part]) -> list[str]:
    """Extracts text content from all TextPart objects in a list of Parts.

    Args:
        parts: A list of `Part` objects.

    Returns:
        A list of strings containing the text content from any `TextPart` objects found.
    """
    return [part.root.text for part in parts if isinstance(part.root, TextPart)]


def get_message_text(message: Message, delimiter='\n') -> str:
    """Extracts and joins all text content from a Message's parts.

    Args:
        message: The `Message` object.
        delimiter: The string to use when joining text from multiple TextParts.

    Returns:
        A single string containing all text content, or an empty string if no text parts are found.
    """
    return delimiter.join(get_text_parts(message.parts))
````

## File: src/a2a/server/events/in_memory_queue_manager.py
````python
import asyncio

from a2a.server.events.event_queue import EventQueue
from a2a.server.events.queue_manager import (
    NoTaskQueue,
    QueueManager,
    TaskQueueExists,
)
from a2a.utils.telemetry import SpanKind, trace_class


@trace_class(kind=SpanKind.SERVER)
class InMemoryQueueManager(QueueManager):
    """InMemoryQueueManager is used for a single binary management.

    This implements the `QueueManager` interface using in-memory storage for event
    queues. It requires all incoming interactions for a given task ID to hit the
    same binary instance.

    This implementation is suitable for single-instance deployments but needs
    a distributed approach for scalable deployments.
    """

    def __init__(self) -> None:
        """Initializes the InMemoryQueueManager."""
        self._task_queue: dict[str, EventQueue] = {}
        self._lock = asyncio.Lock()

    async def add(self, task_id: str, queue: EventQueue):
        """Adds a new event queue for a task ID.

        Raises:
            TaskQueueExists: If a queue for the given `task_id` already exists.
        """
        async with self._lock:
            if task_id in self._task_queue:
                raise TaskQueueExists()
            self._task_queue[task_id] = queue

    async def get(self, task_id: str) -> EventQueue | None:
        """Retrieves the event queue for a task ID.

        Returns:
            The `EventQueue` instance for the `task_id`, or `None` if not found.
        """
        async with self._lock:
            if task_id not in self._task_queue:
                return None
            return self._task_queue[task_id]

    async def tap(self, task_id: str) -> EventQueue | None:
        """Taps the event queue for a task ID to create a child queue.

        Returns:
            A new child `EventQueue` instance, or `None` if the task ID is not found.
        """
        async with self._lock:
            if task_id not in self._task_queue:
                return None
            return self._task_queue[task_id].tap()

    async def close(self, task_id: str):
        """Closes and removes the event queue for a task ID.

        Raises:
            NoTaskQueue: If no queue exists for the given `task_id`.
        """
        async with self._lock:
            if task_id not in self._task_queue:
                raise NoTaskQueue()
            queue = self._task_queue.pop(task_id)
            await queue.close()

    async def create_or_tap(self, task_id: str) -> EventQueue:
        """Creates a new event queue for a task ID if one doesn't exist, otherwise taps the existing one.

        Returns:
            A new or child `EventQueue` instance for the `task_id`.
        """
        async with self._lock:
            if task_id not in self._task_queue:
                queue = EventQueue()
                self._task_queue[task_id] = queue
                return queue
            return self._task_queue[task_id].tap()
````

## File: src/a2a/server/tasks/result_aggregator.py
````python
import asyncio
import logging

from collections.abc import AsyncGenerator, AsyncIterator

from a2a.server.events import Event, EventConsumer
from a2a.server.tasks.task_manager import TaskManager
from a2a.types import Message, Task, TaskState, TaskStatusUpdateEvent


logger = logging.getLogger(__name__)


class ResultAggregator:
    """ResultAggregator is used to process the event streams from an AgentExecutor.

    There are three main ways to use the ResultAggregator:
    1) As part of a processing pipe. consume_and_emit will construct the updated
       task as the events arrive, and re-emit those events for another consumer
    2) As part of a blocking call. consume_all will process the entire stream and
       return the final Task or Message object
    3) As part of a push solution where the latest Task is emitted after processing an event.
       consume_and_emit_task will consume the Event stream, process the events to the current
       Task object and emit that Task object.
    """

    def __init__(self, task_manager: TaskManager):
        """Initializes the ResultAggregator.

        Args:
            task_manager: The `TaskManager` instance to use for processing events
                          and managing the task state.
        """
        self.task_manager = task_manager
        self._message: Message | None = None

    @property
    async def current_result(self) -> Task | Message | None:
        """Returns the current aggregated result (Task or Message).

        This is the latest state processed from the event stream.

        Returns:
            The current `Task` object managed by the `TaskManager`, or the final
            `Message` if one was received, or `None` if no result has been produced yet.
        """
        if self._message:
            return self._message
        return await self.task_manager.get_task()

    async def consume_and_emit(
        self, consumer: EventConsumer
    ) -> AsyncGenerator[Event]:
        """Processes the event stream from the consumer, updates the task state, and re-emits the same events.

        Useful for streaming scenarios where the server needs to observe and
        process events (e.g., save task state, send push notifications) while
        forwarding them to the client.

        Args:
            consumer: The `EventConsumer` to read events from.

        Yields:
            The `Event` objects consumed from the `EventConsumer`.
        """
        async for event in consumer.consume_all():
            await self.task_manager.process(event)
            yield event

    async def consume_all(
        self, consumer: EventConsumer
    ) -> Task | Message | None:
        """Processes the entire event stream from the consumer and returns the final result.

        Blocks until the event stream ends (queue is closed after final event or exception).

        Args:
            consumer: The `EventConsumer` to read events from.

        Returns:
            The final `Task` object or `Message` object after the stream is exhausted.
            Returns `None` if the stream ends without producing a final result.

        Raises:
            BaseException: If the `EventConsumer` raises an exception during consumption.
        """
        async for event in consumer.consume_all():
            if isinstance(event, Message):
                self._message = event
                return event
            await self.task_manager.process(event)
        return await self.task_manager.get_task()

    async def consume_and_break_on_interrupt(
        self, consumer: EventConsumer
    ) -> tuple[Task | Message | None, bool]:
        """Processes the event stream until completion or an interruptable state is encountered.

        Interruptable states currently include `TaskState.auth_required`.
        If interrupted, consumption continues in a background task.

        Args:
            consumer: The `EventConsumer` to read events from.

        Returns:
            A tuple containing:
            - The current aggregated result (`Task` or `Message`) at the point of completion or interruption.
            - A boolean indicating whether the consumption was interrupted (`True`)
              or completed naturally (`False`).

        Raises:
            BaseException: If the `EventConsumer` raises an exception during consumption.
        """
        event_stream = consumer.consume_all()
        interrupted = False
        async for event in event_stream:
            if isinstance(event, Message):
                self._message = event
                return event, False
            await self.task_manager.process(event)
            if (
                isinstance(event, Task | TaskStatusUpdateEvent)
                and event.status.state == TaskState.auth_required
            ):
                # auth-required is a special state: the message should be
                # escalated back to the caller, but the agent is expected to
                # continue producing events once the authorization is received
                # out-of-band. This is in contrast to input-required, where a
                # new request is expected in order for the agent to make progress,
                # so the agent should exit.
                logger.debug(
                    'Encountered an auth-required task: breaking synchronous message/send flow.'
                )
                # TODO: We should track all outstanding tasks to ensure they eventually complete.
                asyncio.create_task(self._continue_consuming(event_stream))
                interrupted = True
                break
        return await self.task_manager.get_task(), interrupted

    async def _continue_consuming(
        self, event_stream: AsyncIterator[Event]
    ) -> None:
        """Continues processing an event stream in a background task.

        Used after an interruptable state (like auth_required) is encountered
        in the synchronous consumption flow.

        Args:
            event_stream: The remaining `AsyncIterator` of events from the consumer.
        """
        async for event in event_stream:
            await self.task_manager.process(event)
````

## File: src/a2a/server/tasks/task_updater.py
````python
import uuid

from typing import Any

from a2a.server.events import EventQueue
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)


class TaskUpdater:
    """Helper class for agents to publish updates to a task's event queue.

    Simplifies the process of creating and enqueueing standard task events.
    """

    def __init__(self, event_queue: EventQueue, task_id: str, context_id: str):
        """Initializes the TaskUpdater.

        Args:
            event_queue: The `EventQueue` associated with the task.
            task_id: The ID of the task.
            context_id: The context ID of the task.
        """
        self.event_queue = event_queue
        self.task_id = task_id
        self.context_id = context_id

    def update_status(
        self, state: TaskState, message: Message | None = None, final=False
    ):
        """Updates the status of the task and publishes a `TaskStatusUpdateEvent`.

        Args:
            state: The new state of the task.
            message: An optional message associated with the status update.
            final: If True, indicates this is the final status update for the task.
        """
        self.event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=self.task_id,
                contextId=self.context_id,
                final=final,
                status=TaskStatus(
                    state=state,
                    message=message,
                ),
            )
        )

    def add_artifact(
        self,
        parts: list[Part],
        artifact_id: str = str(uuid.uuid4()),
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Adds an artifact chunk to the task and publishes a `TaskArtifactUpdateEvent`.

        Args:
            parts: A list of `Part` objects forming the artifact chunk.
            artifact_id: The ID of the artifact. A new UUID is generated if not provided.
            name: Optional name for the artifact.
            metadata: Optional metadata for the artifact.
            append: Optional boolean indicating if this chunk appends to a previous one.
            last_chunk: Optional boolean indicating if this is the last chunk.
        """
        self.event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                taskId=self.task_id,
                contextId=self.context_id,
                artifact=Artifact(
                    artifactId=artifact_id,
                    name=name,
                    parts=parts,
                    metadata=metadata,
                ),
            )
        )

    def complete(self, message: Message | None = None):
        """Marks the task as completed and publishes a final status update."""
        self.update_status(
            TaskState.completed,
            message=message,
            final=True,
        )

    def failed(self, message: Message | None = None):
        """Marks the task as failed and publishes a final status update."""
        self.update_status(TaskState.failed, message=message, final=True)

    def submit(self, message: Message | None = None):
        """Marks the task as submitted and publishes a status update."""
        self.update_status(
            TaskState.submitted,
            message=message,
        )

    def start_work(self, message: Message | None = None):
        """Marks the task as working and publishes a status update."""
        self.update_status(
            TaskState.working,
            message=message,
        )

    def new_agent_message(
        self,
        parts: list[Part],
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Creates a new message object sent by the agent for this task/context.

        Note: This method only *creates* the message object. It does not
              automatically enqueue it.

        Args:
            parts: A list of `Part` objects for the message content.
            final: Optional boolean indicating if this is the final message in a stream.
            metadata: Optional metadata for the message.

        Returns:
            A new `Message` object.
        """
        return Message(
            role=Role.agent,
            taskId=self.task_id,
            contextId=self.context_id,
            messageId=str(uuid.uuid4()),
            metadata=metadata,
            parts=parts,
        )
````

## File: src/a2a/utils/helpers.py
````python
"""General utility functions for the A2A Python SDK."""

import logging

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from a2a.types import (
    Artifact,
    MessageSendParams,
    Part,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TextPart,
)
from a2a.utils.errors import ServerError, UnsupportedOperationError
from a2a.utils.telemetry import trace_function


logger = logging.getLogger(__name__)


@trace_function()
def create_task_obj(message_send_params: MessageSendParams) -> Task:
    """Create a new task object from message send params.

    Generates UUIDs for task and context IDs if they are not already present in the message.

    Args:
        message_send_params: The `MessageSendParams` object containing the initial message.

    Returns:
        A new `Task` object initialized with 'submitted' status and the input message in history.
    """
    if not message_send_params.message.contextId:
        message_send_params.message.contextId = str(uuid4())

    return Task(
        id=str(uuid4()),
        contextId=message_send_params.message.contextId,
        status=TaskStatus(state=TaskState.submitted),
        history=[message_send_params.message],
    )


@trace_function()
def append_artifact_to_task(task: Task, event: TaskArtifactUpdateEvent) -> None:
    """Helper method for updating a Task object with new artifact data from an event.

    Handles creating the artifacts list if it doesn't exist, adding new artifacts,
    and appending parts to existing artifacts based on the `append` flag in the event.

    Args:
        task: The `Task` object to modify.
        event: The `TaskArtifactUpdateEvent` containing the artifact data.
    """
    if not task.artifacts:
        task.artifacts = []

    new_artifact_data: Artifact = event.artifact
    artifact_id: str = new_artifact_data.artifactId
    append_parts: bool = event.append or False

    existing_artifact: Artifact | None = None
    existing_artifact_list_index: int | None = None

    # Find existing artifact by its id
    for i, art in enumerate(task.artifacts):
        if hasattr(art, 'artifactId') and art.artifactId == artifact_id:
            existing_artifact = art
            existing_artifact_list_index = i
            break

    if not append_parts:
        # This represents the first chunk for this artifact index.
        if existing_artifact_list_index is not None:
            # Replace the existing artifact entirely with the new data
            logger.debug(
                f'Replacing artifact at id {artifact_id} for task {task.id}'
            )
            task.artifacts[existing_artifact_list_index] = new_artifact_data
        else:
            # Append the new artifact since no artifact with this index exists yet
            logger.debug(
                f'Adding new artifact with id {artifact_id} for task {task.id}'
            )
            task.artifacts.append(new_artifact_data)
    elif existing_artifact:
        # Append new parts to the existing artifact's part list
        logger.debug(
            f'Appending parts to artifact id {artifact_id} for task {task.id}'
        )
        existing_artifact.parts.extend(new_artifact_data.parts)
    else:
        # We received a chunk to append, but we don't have an existing artifact.
        # we will ignore this chunk
        logger.warning(
            f'Received append=True for nonexistent artifact index {artifact_id} in task {task.id}. Ignoring chunk.'
        )


def build_text_artifact(text: str, artifact_id: str) -> Artifact:
    """Helper to create a text artifact.

    Args:
        text: The text content for the artifact.
        artifact_id: The ID for the artifact.

    Returns:
        An `Artifact` object containing a single `TextPart`.
    """
    text_part = TextPart(text=text)
    part = Part(root=text_part)
    return Artifact(parts=[part], artifactId=artifact_id)


def validate(
    expression: Callable[[Any], bool], error_message: str | None = None
):
    """Decorator that validates if a given expression evaluates to True.

    Typically used on class methods to check capabilities or configuration
    before executing the method's logic. If the expression is False,
    a `ServerError` with an `UnsupportedOperationError` is raised.

    Args:
        expression: A callable that takes the instance (`self`) as its argument
                    and returns a boolean.
        error_message: An optional custom error message for the `UnsupportedOperationError`.
                       If None, the string representation of the expression will be used.
    """

    def decorator(function):
        def wrapper(self, *args, **kwargs):
            if not expression(self):
                final_message = error_message or str(expression)
                logger.error(f'Unsupported Operation: {final_message}')
                raise ServerError(
                    UnsupportedOperationError(message=final_message)
                )
            return function(self, *args, **kwargs)

        return wrapper

    return decorator


def are_modalities_compatible(
    server_output_modes: list[str] | None, client_output_modes: list[str] | None
) -> bool:
    """Checks if server and client output modalities (MIME types) are compatible.

    Modalities are compatible if:
    1. The client specifies no preferred output modes (client_output_modes is None or empty).
    2. The server specifies no supported output modes (server_output_modes is None or empty).
    3. There is at least one common modality between the server's supported list and the client's preferred list.

    Args:
        server_output_modes: A list of MIME types supported by the server/agent for output.
                             Can be None or empty if the server doesn't specify.
        client_output_modes: A list of MIME types preferred by the client for output.
                             Can be None or empty if the client accepts any.

    Returns:
        True if the modalities are compatible, False otherwise.
    """
    if client_output_modes is None or len(client_output_modes) == 0:
        return True

    if server_output_modes is None or len(server_output_modes) == 0:
        return True

    return any(x in server_output_modes for x in client_output_modes)
````

## File: README.md
````markdown
# A2A Python SDK

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![PyPI - Version](https://img.shields.io/pypi/v/a2a-sdk)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/a2a-sdk)

<!-- markdownlint-disable no-inline-html -->

<html>
   <h2 align="center">
   <img src="https://raw.githubusercontent.com/google-a2a/A2A/refs/heads/main/docs/assets/a2a-logo-black.svg" width="256" alt="A2A Logo"/>
   </h2>
   <h3 align="center">A Python library that helps run agentic applications as A2AServers following the <a href="https://google.github.io/A2A">Agent2Agent (A2A) Protocol</a>.</h3>
</html>

<!-- markdownlint-enable no-inline-html -->

## Installation

You can install the A2A SDK using either `uv` or `pip`.

## Prerequisites

- Python 3.10+
- `uv` (optional, but recommended) or `pip`

### Using `uv`

When you're working within a uv project or a virtual environment managed by uv, the preferred way to add packages is using uv add.

```bash
uv add a2a-sdk
```

### Using `pip`

If you prefer to use pip, the standard Python package installer, you can install `a2a-sdk` as follows

```bash
pip install a2a-sdk
```

## Examples

### [Helloworld Example](https://github.com/google/a2a-samples/tree/main/samples/helloworld)

1. Run Remote Agent

   ```bash
   git clone https://github.com/google-a2a/a2a-samples.git
   cd samples/helloworld
   uv run .
   ```

2. In another terminal, run the client

   ```bash
   uv run test_client.py
   ```

You can also find more examples [here](https://github.com/google-a2a/A2A/tree/main/samples/python/agents)

## License

This project is licensed under the terms of the [Apache 2.0 License](https://raw.githubusercontent.com/google-a2a/a2a-python/refs/heads/main/LICENSE).

## Contributing

See [CONTRIBUTING.md](https://github.com/google-a2a/a2a-python/blob/main/CONTRIBUTING.md) for contribution guidelines.
````

## File: src/a2a/utils/task.py
````python
"""Utility functions for creating A2A Task objects."""

import uuid

from a2a.types import Artifact, Message, Task, TaskState, TaskStatus


def new_task(request: Message) -> Task:
    """Creates a new Task object from an initial user message.

    Generates task and context IDs if not provided in the message.

    Args:
        request: The initial `Message` object from the user.

    Returns:
        A new `Task` object initialized with 'submitted' status and the input message in history.
    """
    return Task(
        status=TaskStatus(state=TaskState.submitted),
        id=(request.taskId if request.taskId else str(uuid.uuid4())),
        contextId=(
            request.contextId if request.contextId else str(uuid.uuid4())
        ),
        history=[request],
    )


def completed_task(
    task_id: str,
    context_id: str,
    artifacts: list[Artifact],
    history: list[Message] | None = None,
) -> Task:
    """Creates a Task object in the 'completed' state.

    Useful for constructing a final Task representation when the agent
    finishes and produces artifacts.

    Args:
        task_id: The ID of the task.
        context_id: The context ID of the task.
        artifacts: A list of `Artifact` objects produced by the task.
        history: An optional list of `Message` objects representing the task history.

    Returns:
        A `Task` object with status set to 'completed'.
    """
    if history is None:
        history = []
    return Task(
        status=TaskStatus(state=TaskState.completed),
        id=task_id,
        contextId=context_id,
        artifacts=artifacts,
        history=history,
    )
````

## File: CHANGELOG.md
````markdown
# Changelog

## [0.2.5](https://github.com/google-a2a/a2a-python/compare/v0.2.4...v0.2.5) (2025-05-27)


### Features

* Add a User representation to ServerCallContext ([#116](https://github.com/google-a2a/a2a-python/issues/116)) ([2cc2a0d](https://github.com/google-a2a/a2a-python/commit/2cc2a0de93631aa162823d43fe488173ed8754dc))
* Add functionality for extended agent card.  ([#31](https://github.com/google-a2a/a2a-python/issues/31)) ([20f0826](https://github.com/google-a2a/a2a-python/commit/20f0826a2cb9b77b89b85189fd91e7cd62318a30))
* Introduce a ServerCallContext ([#94](https://github.com/google-a2a/a2a-python/issues/94)) ([85b521d](https://github.com/google-a2a/a2a-python/commit/85b521d8a790dacb775ef764a66fbdd57b180da3))


### Bug Fixes

* fix hello world example for python 3.12 ([#98](https://github.com/google-a2a/a2a-python/issues/98)) ([536e4a1](https://github.com/google-a2a/a2a-python/commit/536e4a11f2f32332968a06e7d0bc4615e047a56c))
* Remove unused dependencies and update py version ([#119](https://github.com/google-a2a/a2a-python/issues/119)) ([9f8bc02](https://github.com/google-a2a/a2a-python/commit/9f8bc023b45544942583818968f3d320e5ff1c3b))
* Update hello world test client to match sdk behavior. Also down-level required python version ([#117](https://github.com/google-a2a/a2a-python/issues/117)) ([04c7c45](https://github.com/google-a2a/a2a-python/commit/04c7c452f5001d69524d94095d11971c1e857f75))
* Update the google adk demos to use ADK v1.0 ([#95](https://github.com/google-a2a/a2a-python/issues/95)) ([c351656](https://github.com/google-a2a/a2a-python/commit/c351656a91c37338668b0cd0c4db5fedd152d743))


### Documentation

* Update README for Python 3.10+ support ([#90](https://github.com/google-a2a/a2a-python/issues/90)) ([e0db20f](https://github.com/google-a2a/a2a-python/commit/e0db20ffc20aa09ee68304cc7e2a67c32ecdd6a8))

## [0.2.4](https://github.com/google-a2a/a2a-python/compare/v0.2.3...v0.2.4) (2025-05-22)

### Features

* Update to support python 3.10 ([#85](https://github.com/google-a2a/a2a-python/issues/85)) ([fd9c3b5](https://github.com/google-a2a/a2a-python/commit/fd9c3b5b0bbef509789a701171d95f690c84750b))


### Bug Fixes

* Throw exception for task_id mismatches ([#70](https://github.com/google-a2a/a2a-python/issues/70)) ([a9781b5](https://github.com/google-a2a/a2a-python/commit/a9781b589075280bfaaab5742d8b950916c9de74))

## [0.2.3](https://github.com/google-a2a/a2a-python/compare/v0.2.2...v0.2.3) (2025-05-20)


### Features

* Add request context builder with referenceTasks ([#56](https://github.com/google-a2a/a2a-python/issues/56)) ([f20bfe7](https://github.com/google-a2a/a2a-python/commit/f20bfe74b8cc854c9c29720b2ea3859aff8f509e))

## [0.2.2](https://github.com/google-a2a/a2a-python/compare/v0.2.1...v0.2.2) (2025-05-20)


### Documentation

* Write/Update Docstrings for Classes/Methods ([#59](https://github.com/google-a2a/a2a-python/issues/59)) ([9f773ef](https://github.com/google-a2a/a2a-python/commit/9f773eff4dddc4eec723d519d0050f21b9ccc042))
````

## File: src/a2a/server/agent_execution/context.py
````python
import uuid

from a2a.server.context import ServerCallContext
from a2a.types import (
    InvalidParamsError,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Task,
)
from a2a.utils import get_message_text
from a2a.utils.errors import ServerError


class RequestContext:
    """Request Context.

    Holds information about the current request being processed by the server,
    including the incoming message, task and context identifiers, and related
    tasks.
    """

    def __init__(
        self,
        request: MessageSendParams | None = None,
        task_id: str | None = None,
        context_id: str | None = None,
        task: Task | None = None,
        related_tasks: list[Task] | None = None,
        call_context: ServerCallContext | None = None,
    ):
        """Initializes the RequestContext.

        Args:
            request: The incoming `MessageSendParams` request payload.
            task_id: The ID of the task explicitly provided in the request or path.
            context_id: The ID of the context explicitly provided in the request or path.
            task: The existing `Task` object retrieved from the store, if any.
            related_tasks: A list of other tasks related to the current request (e.g., for tool use).
        """
        if related_tasks is None:
            related_tasks = []
        self._params = request
        self._task_id = task_id
        self._context_id = context_id
        self._current_task = task
        self._related_tasks = related_tasks
        self._call_context = call_context
        # If the task id and context id were provided, make sure they
        # match the request. Otherwise, create them
        if self._params:
            if task_id:
                self._params.message.taskId = task_id
                if task and task.id != task_id:
                    raise ServerError(InvalidParamsError(message='bad task id'))
            else:
                self._check_or_generate_task_id()
            if context_id:
                self._params.message.contextId = context_id
                if task and task.contextId != context_id:
                    raise ServerError(
                        InvalidParamsError(message='bad context id')
                    )
            else:
                self._check_or_generate_context_id()

    def get_user_input(self, delimiter='\n') -> str:
        """Extracts text content from the user's message parts.

        Args:
            delimiter: The string to use when joining multiple text parts.

        Returns:
            A single string containing all text content from the user message,
            joined by the specified delimiter. Returns an empty string if no
            user message is present or if it contains no text parts.
        """
        if not self._params:
            return ''

        return get_message_text(self._params.message, delimiter)

    def attach_related_task(self, task: Task):
        """Attaches a related task to the context.

        This is useful for scenarios like tool execution where a new task
        might be spawned.

        Args:
            task: The `Task` object to attach.
        """
        self._related_tasks.append(task)

    @property
    def message(self) -> Message | None:
        """The incoming `Message` object from the request, if available."""
        return self._params.message if self._params else None

    @property
    def related_tasks(self) -> list[Task]:
        """A list of tasks related to the current request."""
        return self._related_tasks

    @property
    def current_task(self) -> Task | None:
        """The current `Task` object being processed."""
        return self._current_task

    @current_task.setter
    def current_task(self, task: Task) -> None:
        """Sets the current task object."""
        self._current_task = task

    @property
    def task_id(self) -> str | None:
        """The ID of the task associated with this context."""
        return self._task_id

    @property
    def context_id(self) -> str | None:
        """The ID of the conversation context associated with this task."""
        return self._context_id

    @property
    def configuration(self) -> MessageSendConfiguration | None:
        """The `MessageSendConfiguration` from the request, if available."""
        if not self._params:
            return None
        return self._params.configuration

    @property
    def call_context(self) -> ServerCallContext | None:
        """The server call context associated with this request."""
        return self._call_context

    def _check_or_generate_task_id(self) -> None:
        """Ensures a task ID is present, generating one if necessary."""
        if not self._params:
            return

        if not self._task_id and not self._params.message.taskId:
            self._params.message.taskId = str(uuid.uuid4())
        if self._params.message.taskId:
            self._task_id = self._params.message.taskId

    def _check_or_generate_context_id(self) -> None:
        """Ensures a context ID is present, generating one if necessary."""
        if not self._params:
            return

        if not self._context_id and not self._params.message.contextId:
            self._params.message.contextId = str(uuid.uuid4())
        if self._params.message.contextId:
            self._context_id = self._params.message.contextId
````

## File: src/a2a/server/apps/starlette_app.py
````python
import contextlib
import json
import logging
import traceback

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.authentication import BaseUser
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from a2a.auth.user import UnauthenticatedUser
from a2a.auth.user import User as A2AUser
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers.jsonrpc_handler import JSONRPCHandler
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.types import (
    A2AError,
    A2ARequest,
    AgentCard,
    CancelTaskRequest,
    GetTaskPushNotificationConfigRequest,
    GetTaskRequest,
    InternalError,
    InvalidRequestError,
    JSONParseError,
    JSONRPCError,
    JSONRPCErrorResponse,
    JSONRPCResponse,
    SendMessageRequest,
    SendStreamingMessageRequest,
    SendStreamingMessageResponse,
    SetTaskPushNotificationConfigRequest,
    TaskResubscriptionRequest,
    UnsupportedOperationError,
)
from a2a.utils.errors import MethodNotImplementedError


logger = logging.getLogger(__name__)

# Register Starlette User as an implementation of a2a.auth.user.User
A2AUser.register(BaseUser)


class CallContextBuilder(ABC):
    """A class for building ServerCallContexts using the Starlette Request."""

    @abstractmethod
    def build(self, request: Request) -> ServerCallContext:
        """Builds a ServerCallContext from a Starlette Request."""


class DefaultCallContextBuilder(CallContextBuilder):
    """A default implementation of CallContextBuilder."""

    def build(self, request: Request) -> ServerCallContext:
        user = UnauthenticatedUser()
        state = {}
        with contextlib.suppress(Exception):
            user = request.user
            state['auth'] = request.auth
        return ServerCallContext(user=user, state=state)


class A2AStarletteApplication:
    """A Starlette application implementing the A2A protocol server endpoints.

    Handles incoming JSON-RPC requests, routes them to the appropriate
    handler methods, and manages response generation including Server-Sent Events
    (SSE).
    """

    def __init__(
        self,
        agent_card: AgentCard,
        http_handler: RequestHandler,
        extended_agent_card: AgentCard | None = None,
        context_builder: CallContextBuilder | None = None,
    ):
        """Initializes the A2AStarletteApplication.

        Args:
            agent_card: The AgentCard describing the agent's capabilities.
            http_handler: The handler instance responsible for processing A2A
              requests via http.
            extended_agent_card: An optional, distinct AgentCard to be served
              at the authenticated extended card endpoint.
            context_builder: The CallContextBuilder used to construct the
              ServerCallContext passed to the http_handler. If None, no
              ServerCallContext is passed.
        """
        self.agent_card = agent_card
        self.extended_agent_card = extended_agent_card
        self.handler = JSONRPCHandler(
            agent_card=agent_card, request_handler=http_handler
        )
        if (
            self.agent_card.supportsAuthenticatedExtendedCard
            and self.extended_agent_card is None
        ):
            logger.error(
                'AgentCard.supportsAuthenticatedExtendedCard is True, but no extended_agent_card was provided. The /agent/authenticatedExtendedCard endpoint will return 404.'
            )
        self._context_builder = context_builder or DefaultCallContextBuilder()

    def _generate_error_response(
        self, request_id: str | int | None, error: JSONRPCError | A2AError
    ) -> JSONResponse:
        """Creates a Starlette JSONResponse for a JSON-RPC error.

        Logs the error based on its type.

        Args:
            request_id: The ID of the request that caused the error.
            error: The `JSONRPCError` or `A2AError` object.

        Returns:
            A `JSONResponse` object formatted as a JSON-RPC error response.
        """
        error_resp = JSONRPCErrorResponse(
            id=request_id,
            error=error if isinstance(error, JSONRPCError) else error.root,
        )

        log_level = (
            logging.ERROR
            if not isinstance(error, A2AError)
            or isinstance(error.root, InternalError)
            else logging.WARNING
        )
        logger.log(
            log_level,
            f'Request Error (ID: {request_id}): '
            f"Code={error_resp.error.code}, Message='{error_resp.error.message}'"
            f'{", Data=" + str(error_resp.error.data) if hasattr(error, "data") and error_resp.error.data else ""}',
        )
        return JSONResponse(
            error_resp.model_dump(mode='json', exclude_none=True),
            status_code=200,
        )

    async def _handle_requests(self, request: Request) -> Response:
        """Handles incoming POST requests to the main A2A endpoint.

        Parses the request body as JSON, validates it against A2A request types,
        dispatches it to the appropriate handler method, and returns the response.
        Handles JSON parsing errors, validation errors, and other exceptions,
        returning appropriate JSON-RPC error responses.

        Args:
            request: The incoming Starlette Request object.

        Returns:
            A Starlette Response object (JSONResponse or EventSourceResponse).

        Raises:
            (Implicitly handled): Various exceptions are caught and converted
            into JSON-RPC error responses by this method.
        """
        request_id = None
        body = None

        try:
            body = await request.json()
            a2a_request = A2ARequest.model_validate(body)
            call_context = self._context_builder.build(request)

            request_id = a2a_request.root.id
            request_obj = a2a_request.root

            if isinstance(
                request_obj,
                TaskResubscriptionRequest | SendStreamingMessageRequest,
            ):
                return await self._process_streaming_request(
                    request_id, a2a_request, call_context
                )

            return await self._process_non_streaming_request(
                request_id, a2a_request, call_context
            )
        except MethodNotImplementedError:
            traceback.print_exc()
            return self._generate_error_response(
                request_id, A2AError(root=UnsupportedOperationError())
            )
        except json.decoder.JSONDecodeError as e:
            traceback.print_exc()
            return self._generate_error_response(
                None, A2AError(root=JSONParseError(message=str(e)))
            )
        except ValidationError as e:
            traceback.print_exc()
            return self._generate_error_response(
                request_id,
                A2AError(root=InvalidRequestError(data=json.loads(e.json()))),
            )
        except Exception as e:
            logger.error(f'Unhandled exception: {e}')
            traceback.print_exc()
            return self._generate_error_response(
                request_id, A2AError(root=InternalError(message=str(e)))
            )

    async def _process_streaming_request(
        self,
        request_id: str | int | None,
        a2a_request: A2ARequest,
        context: ServerCallContext,
    ) -> Response:
        """Processes streaming requests (message/stream or tasks/resubscribe).

        Args:
            request_id: The ID of the request.
            a2a_request: The validated A2ARequest object.

        Returns:
            An `EventSourceResponse` object to stream results to the client.
        """
        request_obj = a2a_request.root
        handler_result: Any = None
        if isinstance(
            request_obj,
            SendStreamingMessageRequest,
        ):
            handler_result = self.handler.on_message_send_stream(
                request_obj, context
            )
        elif isinstance(request_obj, TaskResubscriptionRequest):
            handler_result = self.handler.on_resubscribe_to_task(
                request_obj, context
            )

        return self._create_response(handler_result)

    async def _process_non_streaming_request(
        self,
        request_id: str | int | None,
        a2a_request: A2ARequest,
        context: ServerCallContext,
    ) -> Response:
        """Processes non-streaming requests (message/send, tasks/get, tasks/cancel, tasks/pushNotificationConfig/*).

        Args:
            request_id: The ID of the request.
            a2a_request: The validated A2ARequest object.

        Returns:
            A `JSONResponse` object containing the result or error.
        """
        request_obj = a2a_request.root
        handler_result: Any = None
        match request_obj:
            case SendMessageRequest():
                handler_result = await self.handler.on_message_send(
                    request_obj, context
                )
            case CancelTaskRequest():
                handler_result = await self.handler.on_cancel_task(
                    request_obj, context
                )
            case GetTaskRequest():
                handler_result = await self.handler.on_get_task(
                    request_obj, context
                )
            case SetTaskPushNotificationConfigRequest():
                handler_result = await self.handler.set_push_notification(
                    request_obj,
                    context,
                )
            case GetTaskPushNotificationConfigRequest():
                handler_result = await self.handler.get_push_notification(
                    request_obj,
                    context,
                )
            case _:
                logger.error(
                    f'Unhandled validated request type: {type(request_obj)}'
                )
                error = UnsupportedOperationError(
                    message=f'Request type {type(request_obj).__name__} is unknown.'
                )
                handler_result = JSONRPCErrorResponse(
                    id=request_id, error=error
                )

        return self._create_response(handler_result)

    def _create_response(
        self,
        handler_result: (
            AsyncGenerator[SendStreamingMessageResponse]
            | JSONRPCErrorResponse
            | JSONRPCResponse
        ),
    ) -> Response:
        """Creates a Starlette Response based on the result from the request handler.

        Handles:
        - AsyncGenerator for Server-Sent Events (SSE).
        - JSONRPCErrorResponse for explicit errors returned by handlers.
        - Pydantic RootModels (like GetTaskResponse) containing success or error
        payloads.

        Args:
            handler_result: The result from a request handler method. Can be an
                async generator for streaming or a Pydantic model for non-streaming.

        Returns:
            A Starlette JSONResponse or EventSourceResponse.
        """
        if isinstance(handler_result, AsyncGenerator):
            # Result is a stream of SendStreamingMessageResponse objects
            async def event_generator(
                stream: AsyncGenerator[SendStreamingMessageResponse],
            ) -> AsyncGenerator[dict[str, str]]:
                async for item in stream:
                    yield {'data': item.root.model_dump_json(exclude_none=True)}

            return EventSourceResponse(event_generator(handler_result))
        if isinstance(handler_result, JSONRPCErrorResponse):
            return JSONResponse(
                handler_result.model_dump(
                    mode='json',
                    exclude_none=True,
                )
            )

        return JSONResponse(
            handler_result.root.model_dump(mode='json', exclude_none=True)
        )

    async def _handle_get_agent_card(self, request: Request) -> JSONResponse:
        """Handles GET requests for the agent card endpoint.

        Args:
            request: The incoming Starlette Request object.

        Returns:
            A JSONResponse containing the agent card data.
        """
        # The public agent card is a direct serialization of the agent_card
        # provided at initialization.
        return JSONResponse(
            self.agent_card.model_dump(mode='json', exclude_none=True)
        )

    async def _handle_get_authenticated_extended_agent_card(
        self, request: Request
    ) -> JSONResponse:
        """Handles GET requests for the authenticated extended agent card."""
        if not self.agent_card.supportsAuthenticatedExtendedCard:
            return JSONResponse(
                {'error': 'Extended agent card not supported or not enabled.'},
                status_code=404,
            )

        # If an explicit extended_agent_card is provided, serve that.
        if self.extended_agent_card:
            return JSONResponse(
                self.extended_agent_card.model_dump(
                    mode='json', exclude_none=True
                )
            )
        # If supportsAuthenticatedExtendedCard is true, but no specific
        # extended_agent_card was provided during server initialization,
        # return a 404
        return JSONResponse(
            {
                'error': 'Authenticated extended agent card is supported but not configured on the server.'
            },
            status_code=404,
        )

    def routes(
        self,
        agent_card_url: str = '/.well-known/agent.json',
        extended_agent_card_url: str = '/agent/authenticatedExtendedCard',
        rpc_url: str = '/',
    ) -> list[Route]:
        """Returns the Starlette Routes for handling A2A requests.

        Args:
            agent_card_url: The URL path for the agent card endpoint.
            rpc_url: The URL path for the A2A JSON-RPC endpoint (POST requests).
            extended_agent_card_url: The URL for the authenticated extended agent card endpoint.

        Returns:
            A list of Starlette Route objects.
        """
        app_routes = [
            Route(
                rpc_url,
                self._handle_requests,
                methods=['POST'],
                name='a2a_handler',
            ),
            Route(
                agent_card_url,
                self._handle_get_agent_card,
                methods=['GET'],
                name='agent_card',
            ),
        ]

        if self.agent_card.supportsAuthenticatedExtendedCard:
            app_routes.append(
                Route(
                    extended_agent_card_url,
                    self._handle_get_authenticated_extended_agent_card,
                    methods=['GET'],
                    name='authenticated_extended_agent_card',
                )
            )
        return app_routes

    def build(
        self,
        agent_card_url: str = '/.well-known/agent.json',
        extended_agent_card_url: str = '/agent/authenticatedExtendedCard',
        rpc_url: str = '/',
        **kwargs: Any,
    ) -> Starlette:
        """Builds and returns the Starlette application instance.

        Args:
            agent_card_url: The URL path for the agent card endpoint.
            rpc_url: The URL path for the A2A JSON-RPC endpoint (POST requests).
            extended_agent_card_url: The URL for the authenticated extended agent card endpoint.
            **kwargs: Additional keyword arguments to pass to the Starlette
              constructor.

        Returns:
            A configured Starlette application instance.
        """
        app_routes = self.routes(
            agent_card_url, extended_agent_card_url, rpc_url
        )
        if 'routes' in kwargs:
            kwargs['routes'].extend(app_routes)
        else:
            kwargs['routes'] = app_routes

        return Starlette(**kwargs)
````

## File: src/a2a/utils/__init__.py
````python
"""Utility functions for the A2A Python SDK."""

from a2a.utils.artifact import (
    new_artifact,
    new_data_artifact,
    new_text_artifact,
)
from a2a.utils.helpers import (
    append_artifact_to_task,
    are_modalities_compatible,
    build_text_artifact,
    create_task_obj,
)
from a2a.utils.message import (
    get_message_text,
    get_text_parts,
    new_agent_parts_message,
    new_agent_text_message,
)
from a2a.utils.task import (
    completed_task,
    new_task,
)


__all__ = [
    'append_artifact_to_task',
    'are_modalities_compatible',
    'build_text_artifact',
    'completed_task',
    'create_task_obj',
    'get_message_text',
    'get_text_parts',
    'new_agent_parts_message',
    'new_agent_text_message',
    'new_artifact',
    'new_data_artifact',
    'new_task',
    'new_text_artifact',
]
````

## File: src/a2a/server/events/event_consumer.py
````python
import asyncio
import logging
import sys

from collections.abc import AsyncGenerator

from a2a.server.events.event_queue import Event, EventQueue
from a2a.types import (
    InternalError,
    Message,
    Task,
    TaskState,
    TaskStatusUpdateEvent,
)
from a2a.utils.errors import ServerError
from a2a.utils.telemetry import SpanKind, trace_class


# This is an alias to the exception for closed queue
QueueClosed = asyncio.QueueEmpty

# When using python 3.13 or higher, the closed queue signal is QueueShutdown
if sys.version_info >= (3, 13):
    QueueClosed = asyncio.QueueShutDown

logger = logging.getLogger(__name__)


@trace_class(kind=SpanKind.SERVER)
class EventConsumer:
    """Consumer to read events from the agent event queue."""

    def __init__(self, queue: EventQueue):
        """Initializes the EventConsumer.

        Args:
            queue: The `EventQueue` instance to consume events from.
        """
        self.queue = queue
        self._timeout = 0.5
        self._exception: BaseException | None = None
        logger.debug('EventConsumer initialized')

    async def consume_one(self) -> Event:
        """Consume one event from the agent event queue non-blocking.

        Returns:
            The next event from the queue.

        Raises:
            ServerError: If the queue is empty when attempting to dequeue
                immediately.
        """
        logger.debug('Attempting to consume one event.')
        try:
            event = await self.queue.dequeue_event(no_wait=True)
        except asyncio.QueueEmpty as e:
            logger.warning('Event queue was empty in consume_one.')
            raise ServerError(
                InternalError(message='Agent did not return any response')
            ) from e

        logger.debug(f'Dequeued event of type: {type(event)} in consume_one.')

        self.queue.task_done()

        return event

    async def consume_all(self) -> AsyncGenerator[Event]:
        """Consume all the generated streaming events from the agent.

        This method yields events as they become available from the queue
        until a final event is received or the queue is closed. It also
        monitors for exceptions set by the `agent_task_callback`.

        Yields:
            Events dequeued from the queue.

        Raises:
            BaseException: If an exception was set by the `agent_task_callback`.
        """
        logger.debug('Starting to consume all events from the queue.')
        while True:
            if self._exception:
                raise self._exception
            try:
                # We use a timeout when waiting for an event from the queue.
                # This is required because it allows the loop to check if
                # `self._exception` has been set by the `agent_task_callback`.
                # Without the timeout, loop might hang indefinitely if no events are
                # enqueued by the agent and the agent simply threw an exception
                event = await asyncio.wait_for(
                    self.queue.dequeue_event(), timeout=self._timeout
                )
                logger.debug(
                    f'Dequeued event of type: {type(event)} in consume_all.'
                )
                self.queue.task_done()
                logger.debug(
                    'Marked task as done in event queue in consume_all'
                )

                is_final_event = (
                    (isinstance(event, TaskStatusUpdateEvent) and event.final)
                    or isinstance(event, Message)
                    or (
                        isinstance(event, Task)
                        and event.status.state
                        in (
                            TaskState.completed,
                            TaskState.canceled,
                            TaskState.failed,
                            TaskState.rejected,
                            TaskState.unknown,
                        )
                    )
                )

                # Make sure the yield is after the close events, otherwise
                # the caller may end up in a blocked state where this
                # generator isn't called again to close things out and the
                # other part is waiting for an event or a closed queue.
                if is_final_event:
                    logger.debug('Stopping event consumption in consume_all.')
                    await self.queue.close()
                    yield event
                    break
                yield event
            except TimeoutError:
                # continue polling until there is a final event
                continue
            except QueueClosed:
                # Confirm that the queue is closed, e.g. we aren't on
                # python 3.12 and get a queue empty error on an open queue
                if self.queue.is_closed():
                    break

    def agent_task_callback(self, agent_task: asyncio.Task[None]):
        """Callback to handle exceptions from the agent's execution task.

        If the agent's asyncio task raises an exception, this callback is
        invoked, and the exception is stored to be re-raised by the consumer loop.

        Args:
            agent_task: The asyncio.Task that completed.
        """
        logger.debug('Agent task callback triggered.')
        if agent_task.exception() is not None:
            self._exception = agent_task.exception()
````

## File: tests/server/request_handlers/test_jsonrpc_handler.py
````python
import unittest
import unittest.async_case
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.agent_execution.request_context_builder import (
    RequestContextBuilder,
)
from a2a.server.context import ServerCallContext
from a2a.server.events import QueueManager
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler, JSONRPCHandler
from a2a.server.tasks import InMemoryPushNotifier, PushNotifier, TaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    Artifact,
    CancelTaskRequest,
    CancelTaskSuccessResponse,
    GetTaskPushNotificationConfigRequest,
    GetTaskPushNotificationConfigResponse,
    GetTaskPushNotificationConfigSuccessResponse,
    GetTaskRequest,
    GetTaskResponse,
    GetTaskSuccessResponse,
    InternalError,
    JSONRPCErrorResponse,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    PushNotificationConfig,
    SendMessageRequest,
    SendMessageSuccessResponse,
    SendStreamingMessageRequest,
    SendStreamingMessageSuccessResponse,
    SetTaskPushNotificationConfigRequest,
    SetTaskPushNotificationConfigResponse,
    SetTaskPushNotificationConfigSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskNotFoundError,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskResubscriptionRequest,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError

MINIMAL_TASK: dict[str, Any] = {
    'id': 'task_123',
    'contextId': 'session-xyz',
    'status': {'state': 'submitted'},
    'kind': 'task',
}
MESSAGE_PAYLOAD: dict[str, Any] = {
    'role': 'agent',
    'parts': [{'text': 'test message'}],
    'messageId': '111',
}


class TestJSONRPCtHandler(unittest.async_case.IsolatedAsyncioTestCase):
    @pytest.fixture(autouse=True)
    def init_fixtures(self) -> None:
        self.mock_agent_card = MagicMock(
            spec=AgentCard, url='http://agent.example.com/api'
        )

    async def test_on_get_task_success(self) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        call_context = ServerCallContext(state={'foo': 'bar'})
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        task_id = 'test_task_id'
        mock_task = Task(**MINIMAL_TASK)
        mock_task_store.get.return_value = mock_task
        request = GetTaskRequest(id='1', params=TaskQueryParams(id=task_id))
        response: GetTaskResponse = await handler.on_get_task(
            request, call_context
        )
        self.assertIsInstance(response.root, GetTaskSuccessResponse)
        assert response.root.result == mock_task  # type: ignore
        mock_task_store.get.assert_called_once_with(task_id)

    async def test_on_get_task_not_found(self) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        mock_task_store.get.return_value = None
        request = GetTaskRequest(
            id='1',
            method='tasks/get',
            params=TaskQueryParams(id='nonexistent_id'),
        )
        call_context = ServerCallContext(state={'foo': 'bar'})
        response: GetTaskResponse = await handler.on_get_task(
            request, call_context
        )
        self.assertIsInstance(response.root, JSONRPCErrorResponse)
        assert response.root.error == TaskNotFoundError()  # type: ignore

    async def test_on_cancel_task_success(self) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        task_id = 'test_task_id'
        mock_task = Task(**MINIMAL_TASK)
        mock_task_store.get.return_value = mock_task
        mock_agent_executor.cancel.return_value = None
        call_context = ServerCallContext(state={'foo': 'bar'})

        async def streaming_coro():
            yield mock_task

        with patch(
            'a2a.server.request_handlers.default_request_handler.EventConsumer.consume_all',
            return_value=streaming_coro(),
        ):
            request = CancelTaskRequest(id='1', params=TaskIdParams(id=task_id))
            response = await handler.on_cancel_task(request, call_context)
            assert mock_agent_executor.cancel.call_count == 1
            self.assertIsInstance(response.root, CancelTaskSuccessResponse)
            assert response.root.result == mock_task  # type: ignore
            mock_agent_executor.cancel.assert_called_once()

    async def test_on_cancel_task_not_supported(self) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        task_id = 'test_task_id'
        mock_task = Task(**MINIMAL_TASK)
        mock_task_store.get.return_value = mock_task
        mock_agent_executor.cancel.return_value = None
        call_context = ServerCallContext(state={'foo': 'bar'})

        async def streaming_coro():
            raise ServerError(UnsupportedOperationError())
            yield

        with patch(
            'a2a.server.request_handlers.default_request_handler.EventConsumer.consume_all',
            return_value=streaming_coro(),
        ):
            request = CancelTaskRequest(id='1', params=TaskIdParams(id=task_id))
            response = await handler.on_cancel_task(request, call_context)
            assert mock_agent_executor.cancel.call_count == 1
            self.assertIsInstance(response.root, JSONRPCErrorResponse)
            assert response.root.error == UnsupportedOperationError()  # type: ignore
            mock_agent_executor.cancel.assert_called_once()

    async def test_on_cancel_task_not_found(self) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        mock_task_store.get.return_value = None
        request = CancelTaskRequest(
            id='1',
            method='tasks/cancel',
            params=TaskIdParams(id='nonexistent_id'),
        )
        response = await handler.on_cancel_task(request)
        self.assertIsInstance(response.root, JSONRPCErrorResponse)
        assert response.root.error == TaskNotFoundError()  # type: ignore
        mock_task_store.get.assert_called_once_with('nonexistent_id')
        mock_agent_executor.cancel.assert_not_called()

    @patch(
        'a2a.server.agent_execution.simple_request_context_builder.SimpleRequestContextBuilder.build'
    )
    async def test_on_message_new_message_success(
        self, _mock_builder_build: AsyncMock
    ) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        mock_task = Task(**MINIMAL_TASK)
        mock_task_store.get.return_value = mock_task
        mock_agent_executor.execute.return_value = None

        _mock_builder_build.return_value = RequestContext(
            request=MagicMock(),
            task_id='task_123',
            context_id='session-xyz',
            task=None,
            related_tasks=None,
        )

        async def streaming_coro():
            yield mock_task

        with patch(
            'a2a.server.request_handlers.default_request_handler.EventConsumer.consume_all',
            return_value=streaming_coro(),
        ):
            request = SendMessageRequest(
                id='1',
                params=MessageSendParams(message=Message(**MESSAGE_PAYLOAD)),
            )
            response = await handler.on_message_send(request)
            assert mock_agent_executor.execute.call_count == 1
            self.assertIsInstance(response.root, SendMessageSuccessResponse)
            assert response.root.result == mock_task  # type: ignore
            mock_agent_executor.execute.assert_called_once()

    async def test_on_message_new_message_with_existing_task_success(
        self,
    ) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        mock_task = Task(**MINIMAL_TASK)
        mock_task_store.get.return_value = mock_task
        mock_agent_executor.execute.return_value = None

        async def streaming_coro():
            yield mock_task

        with patch(
            'a2a.server.request_handlers.default_request_handler.EventConsumer.consume_all',
            return_value=streaming_coro(),
        ):
            request = SendMessageRequest(
                id='1',
                params=MessageSendParams(
                    message=Message(
                        **MESSAGE_PAYLOAD,
                        taskId=mock_task.id,
                        contextId=mock_task.contextId,
                    )
                ),
            )
            response = await handler.on_message_send(request)
            assert mock_agent_executor.execute.call_count == 1
            self.assertIsInstance(response.root, SendMessageSuccessResponse)
            assert response.root.result == mock_task  # type: ignore
            mock_agent_executor.execute.assert_called_once()

    async def test_on_message_error(self) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        mock_task_store.get.return_value = None
        mock_agent_executor.execute.return_value = None

        async def streaming_coro():
            raise ServerError(error=UnsupportedOperationError())
            yield

        with patch(
            'a2a.server.request_handlers.default_request_handler.EventConsumer.consume_all',
            return_value=streaming_coro(),
        ):
            request = SendMessageRequest(
                id='1',
                params=MessageSendParams(
                    message=Message(
                        **MESSAGE_PAYLOAD,
                    )
                ),
            )
            response = await handler.on_message_send(request)

            self.assertIsInstance(response.root, JSONRPCErrorResponse)
            assert response.root.error == UnsupportedOperationError()  # type: ignore
            mock_agent_executor.execute.assert_called_once()

    @patch(
        'a2a.server.agent_execution.simple_request_context_builder.SimpleRequestContextBuilder.build'
    )
    async def test_on_message_stream_new_message_success(
        self, _mock_builder_build: AsyncMock
    ) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )

        self.mock_agent_card.capabilities = AgentCapabilities(streaming=True)
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        _mock_builder_build.return_value = RequestContext(
            request=MagicMock(),
            task_id='task_123',
            context_id='session-xyz',
            task=None,
            related_tasks=None,
        )

        events: list[Any] = [
            Task(**MINIMAL_TASK),
            TaskArtifactUpdateEvent(
                taskId='task_123',
                contextId='session-xyz',
                artifact=Artifact(
                    artifactId='11', parts=[Part(TextPart(text='text'))]
                ),
            ),
            TaskStatusUpdateEvent(
                taskId='task_123',
                contextId='session-xyz',
                status=TaskStatus(state=TaskState.completed),
                final=True,
            ),
        ]

        async def streaming_coro():
            for event in events:
                yield event

        with patch(
            'a2a.server.request_handlers.default_request_handler.EventConsumer.consume_all',
            return_value=streaming_coro(),
        ):
            mock_task_store.get.return_value = None
            mock_agent_executor.execute.return_value = None
            request = SendStreamingMessageRequest(
                id='1',
                params=MessageSendParams(message=Message(**MESSAGE_PAYLOAD)),
            )
            response = handler.on_message_send_stream(request)
            assert isinstance(response, AsyncGenerator)
            collected_events: list[Any] = []
            async for event in response:
                collected_events.append(event)
            assert len(collected_events) == len(events)
            for i, event in enumerate(collected_events):
                assert isinstance(
                    event.root, SendStreamingMessageSuccessResponse
                )
                assert event.root.result == events[i]
            mock_agent_executor.execute.assert_called_once()

    async def test_on_message_stream_new_message_existing_task_success(
        self,
    ) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )

        self.mock_agent_card.capabilities = AgentCapabilities(streaming=True)

        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        mock_task = Task(**MINIMAL_TASK, history=[])
        events: list[Any] = [
            mock_task,
            TaskArtifactUpdateEvent(
                taskId='task_123',
                contextId='session-xyz',
                artifact=Artifact(
                    artifactId='11', parts=[Part(TextPart(text='text'))]
                ),
            ),
            TaskStatusUpdateEvent(
                taskId='task_123',
                contextId='session-xyz',
                status=TaskStatus(state=TaskState.working),
                final=True,
            ),
        ]

        async def streaming_coro():
            for event in events:
                yield event

        with patch(
            'a2a.server.request_handlers.default_request_handler.EventConsumer.consume_all',
            return_value=streaming_coro(),
        ):
            mock_task_store.get.return_value = mock_task
            mock_agent_executor.execute.return_value = None
            request = SendStreamingMessageRequest(
                id='1',
                params=MessageSendParams(
                    message=Message(
                        **MESSAGE_PAYLOAD,
                        taskId=mock_task.id,
                        contextId=mock_task.contextId,
                    )
                ),
            )
            response = handler.on_message_send_stream(request)
            assert isinstance(response, AsyncGenerator)
            collected_events = [item async for item in response]
            assert len(collected_events) == len(events)
            mock_agent_executor.execute.assert_called_once()
            assert mock_task.history is not None and len(mock_task.history) == 1

    async def test_set_push_notification_success(self) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        mock_push_notifier = AsyncMock(spec=PushNotifier)
        request_handler = DefaultRequestHandler(
            mock_agent_executor,
            mock_task_store,
            push_notifier=mock_push_notifier,
        )
        self.mock_agent_card.capabilities = AgentCapabilities(
            streaming=True, pushNotifications=True
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        mock_task = Task(**MINIMAL_TASK)
        mock_task_store.get.return_value = mock_task
        task_push_config = TaskPushNotificationConfig(
            taskId=mock_task.id,
            pushNotificationConfig=PushNotificationConfig(
                url='http://example.com'
            ),
        )
        request = SetTaskPushNotificationConfigRequest(
            id='1', params=task_push_config
        )
        response: SetTaskPushNotificationConfigResponse = (
            await handler.set_push_notification(request)
        )
        self.assertIsInstance(
            response.root, SetTaskPushNotificationConfigSuccessResponse
        )
        assert response.root.result == task_push_config  # type: ignore
        mock_push_notifier.set_info.assert_called_once_with(
            mock_task.id, task_push_config.pushNotificationConfig
        )

    async def test_get_push_notification_success(self) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        mock_httpx_client = AsyncMock(spec=httpx.AsyncClient)
        push_notifier = InMemoryPushNotifier(httpx_client=mock_httpx_client)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store, push_notifier=push_notifier
        )
        self.mock_agent_card.capabilities = AgentCapabilities(
            streaming=True, pushNotifications=True
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        mock_task = Task(**MINIMAL_TASK)
        mock_task_store.get.return_value = mock_task
        task_push_config = TaskPushNotificationConfig(
            taskId=mock_task.id,
            pushNotificationConfig=PushNotificationConfig(
                url='http://example.com'
            ),
        )
        request = SetTaskPushNotificationConfigRequest(
            id='1', params=task_push_config
        )
        await handler.set_push_notification(request)

        get_request: GetTaskPushNotificationConfigRequest = (
            GetTaskPushNotificationConfigRequest(
                id='1', params=TaskIdParams(id=mock_task.id)
            )
        )
        get_response: GetTaskPushNotificationConfigResponse = (
            await handler.get_push_notification(get_request)
        )
        self.assertIsInstance(
            get_response.root, GetTaskPushNotificationConfigSuccessResponse
        )
        assert get_response.root.result == task_push_config  # type: ignore

    @patch(
        'a2a.server.agent_execution.simple_request_context_builder.SimpleRequestContextBuilder.build'
    )
    async def test_on_message_stream_new_message_send_push_notification_success(
        self, _mock_builder_build: AsyncMock
    ) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        mock_httpx_client = AsyncMock(spec=httpx.AsyncClient)
        push_notifier = InMemoryPushNotifier(httpx_client=mock_httpx_client)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store, push_notifier=push_notifier
        )
        self.mock_agent_card.capabilities = AgentCapabilities(
            streaming=True, pushNotifications=True
        )
        _mock_builder_build.return_value = RequestContext(
            request=MagicMock(),
            task_id='task_123',
            context_id='session-xyz',
            task=None,
            related_tasks=None,
        )

        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        events: list[Any] = [
            Task(**MINIMAL_TASK),
            TaskArtifactUpdateEvent(
                taskId='task_123',
                contextId='session-xyz',
                artifact=Artifact(
                    artifactId='11', parts=[Part(TextPart(text='text'))]
                ),
            ),
            TaskStatusUpdateEvent(
                taskId='task_123',
                contextId='session-xyz',
                status=TaskStatus(state=TaskState.completed),
                final=True,
            ),
        ]

        async def streaming_coro():
            for event in events:
                yield event

        with patch(
            'a2a.server.request_handlers.default_request_handler.EventConsumer.consume_all',
            return_value=streaming_coro(),
        ):
            mock_task_store.get.return_value = None
            mock_agent_executor.execute.return_value = None
            mock_httpx_client.post.return_value = httpx.Response(200)
            request = SendStreamingMessageRequest(
                id='1',
                params=MessageSendParams(message=Message(**MESSAGE_PAYLOAD)),
            )
            request.params.configuration = MessageSendConfiguration(
                acceptedOutputModes=['text'],
                pushNotificationConfig=PushNotificationConfig(
                    url='http://example.com'
                ),
            )
            response = handler.on_message_send_stream(request)
            assert isinstance(response, AsyncGenerator)

            collected_events = [item async for item in response]
            assert len(collected_events) == len(events)

            calls = [
                call(
                    'http://example.com',
                    json={
                        'contextId': 'session-xyz',
                        'id': 'task_123',
                        'kind': 'task',
                        'status': {'state': 'submitted'},
                    },
                ),
                call(
                    'http://example.com',
                    json={
                        'artifacts': [
                            {
                                'artifactId': '11',
                                'parts': [
                                    {
                                        'kind': 'text',
                                        'text': 'text',
                                    }
                                ],
                            }
                        ],
                        'contextId': 'session-xyz',
                        'id': 'task_123',
                        'kind': 'task',
                        'status': {'state': 'submitted'},
                    },
                ),
                call(
                    'http://example.com',
                    json={
                        'artifacts': [
                            {
                                'artifactId': '11',
                                'parts': [
                                    {
                                        'kind': 'text',
                                        'text': 'text',
                                    }
                                ],
                            }
                        ],
                        'contextId': 'session-xyz',
                        'id': 'task_123',
                        'kind': 'task',
                        'status': {'state': 'completed'},
                    },
                ),
            ]
            mock_httpx_client.post.assert_has_calls(calls)

    async def test_on_resubscribe_existing_task_success(
        self,
    ) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        mock_queue_manager = AsyncMock(spec=QueueManager)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store, mock_queue_manager
        )
        self.mock_agent_card = MagicMock(spec=AgentCard)
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        mock_task = Task(**MINIMAL_TASK, history=[])
        events: list[Any] = [
            TaskArtifactUpdateEvent(
                taskId='task_123',
                contextId='session-xyz',
                artifact=Artifact(
                    artifactId='11', parts=[Part(TextPart(text='text'))]
                ),
            ),
            TaskStatusUpdateEvent(
                taskId='task_123',
                contextId='session-xyz',
                status=TaskStatus(state=TaskState.completed),
                final=True,
            ),
        ]

        async def streaming_coro():
            for event in events:
                yield event

        with patch(
            'a2a.server.request_handlers.default_request_handler.EventConsumer.consume_all',
            return_value=streaming_coro(),
        ):
            mock_task_store.get.return_value = mock_task
            mock_queue_manager.tap.return_value = EventQueue()
            request = TaskResubscriptionRequest(
                id='1', params=TaskIdParams(id=mock_task.id)
            )
            response = handler.on_resubscribe_to_task(request)
            assert isinstance(response, AsyncGenerator)
            collected_events: list[Any] = []
            async for event in response:
                collected_events.append(event)
            assert len(collected_events) == len(events)
            assert mock_task.history is not None and len(mock_task.history) == 0

    async def test_on_resubscribe_no_existing_task_error(self) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        mock_task_store.get.return_value = None
        request = TaskResubscriptionRequest(
            id='1', params=TaskIdParams(id='nonexistent_id')
        )
        response = handler.on_resubscribe_to_task(request)
        assert isinstance(response, AsyncGenerator)
        collected_events: list[Any] = []
        async for event in response:
            collected_events.append(event)
        assert len(collected_events) == 1
        self.assertIsInstance(collected_events[0].root, JSONRPCErrorResponse)
        assert collected_events[0].root.error == TaskNotFoundError()

    async def test_streaming_not_supported_error(
        self,
    ) -> None:
        """Test that on_message_send_stream raises an error when streaming not supported."""
        # Arrange
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        # Create agent card with streaming capability disabled
        self.mock_agent_card.capabilities = AgentCapabilities(streaming=False)
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)

        # Act & Assert
        request = SendStreamingMessageRequest(
            id='1',
            params=MessageSendParams(message=Message(**MESSAGE_PAYLOAD)),
        )

        # Should raise ServerError about streaming not supported
        with self.assertRaises(ServerError) as context:
            async for _ in handler.on_message_send_stream(request):
                pass

        aaa = context.exception
        self.assertEqual(
            str(context.exception.error.message),
            'Streaming is not supported by the agent',
        )

    async def test_push_notifications_not_supported_error(self) -> None:
        """Test that set_push_notification raises an error when push notifications not supported."""
        # Arrange
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        # Create agent card with push notifications capability disabled
        self.mock_agent_card.capabilities = AgentCapabilities(
            pushNotifications=False, streaming=True
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)

        # Act & Assert
        task_push_config = TaskPushNotificationConfig(
            taskId='task_123',
            pushNotificationConfig=PushNotificationConfig(
                url='http://example.com'
            ),
        )
        request = SetTaskPushNotificationConfigRequest(
            id='1', params=task_push_config
        )

        # Should raise ServerError about push notifications not supported
        with self.assertRaises(ServerError) as context:
            await handler.set_push_notification(request)

        self.assertEqual(
            str(context.exception.error.message),
            'Push notifications are not supported by the agent',
        )

    async def test_on_get_push_notification_no_push_notifier(self) -> None:
        """Test get_push_notification with no push notifier configured."""
        # Arrange
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        # Create request handler without a push notifier
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        self.mock_agent_card.capabilities = AgentCapabilities(
            pushNotifications=True
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)

        mock_task = Task(**MINIMAL_TASK)
        mock_task_store.get.return_value = mock_task

        # Act
        get_request = GetTaskPushNotificationConfigRequest(
            id='1', params=TaskIdParams(id=mock_task.id)
        )
        response = await handler.get_push_notification(get_request)

        # Assert
        self.assertIsInstance(response.root, JSONRPCErrorResponse)
        self.assertEqual(response.root.error, UnsupportedOperationError())  # type: ignore

    async def test_on_set_push_notification_no_push_notifier(self) -> None:
        """Test set_push_notification with no push notifier configured."""
        # Arrange
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        # Create request handler without a push notifier
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        self.mock_agent_card.capabilities = AgentCapabilities(
            pushNotifications=True
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)

        mock_task = Task(**MINIMAL_TASK)
        mock_task_store.get.return_value = mock_task

        # Act
        task_push_config = TaskPushNotificationConfig(
            taskId=mock_task.id,
            pushNotificationConfig=PushNotificationConfig(
                url='http://example.com'
            ),
        )
        request = SetTaskPushNotificationConfigRequest(
            id='1', params=task_push_config
        )
        response = await handler.set_push_notification(request)

        # Assert
        self.assertIsInstance(response.root, JSONRPCErrorResponse)
        self.assertEqual(response.root.error, UnsupportedOperationError())  # type: ignore

    async def test_on_message_send_internal_error(self) -> None:
        """Test on_message_send with an internal error."""
        # Arrange
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)

        # Make the request handler raise an Internal error without specifying an error type
        async def raise_server_error(*args, **kwargs):
            raise ServerError(InternalError(message='Internal Error'))

        # Patch the method to raise an error
        with patch.object(
            request_handler, 'on_message_send', side_effect=raise_server_error
        ):
            # Act
            request = SendMessageRequest(
                id='1',
                params=MessageSendParams(message=Message(**MESSAGE_PAYLOAD)),
            )
            response = await handler.on_message_send(request)

            # Assert
            self.assertIsInstance(response.root, JSONRPCErrorResponse)
            self.assertIsInstance(response.root.error, InternalError)  # type: ignore

    async def test_on_message_stream_internal_error(self) -> None:
        """Test on_message_send_stream with an internal error."""
        # Arrange
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        self.mock_agent_card.capabilities = AgentCapabilities(streaming=True)
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)

        # Make the request handler raise an Internal error without specifying an error type
        async def raise_server_error(*args, **kwargs):
            raise ServerError(InternalError(message='Internal Error'))
            yield  # Need this to make it an async generator

        # Patch the method to raise an error
        with patch.object(
            request_handler,
            'on_message_send_stream',
            return_value=raise_server_error(),
        ):
            # Act
            request = SendStreamingMessageRequest(
                id='1',
                params=MessageSendParams(message=Message(**MESSAGE_PAYLOAD)),
            )

            # Get the single error response
            responses = []
            async for response in handler.on_message_send_stream(request):
                responses.append(response)

            # Assert
            self.assertEqual(len(responses), 1)
            self.assertIsInstance(responses[0].root, JSONRPCErrorResponse)
            self.assertIsInstance(responses[0].root.error, InternalError)

    async def test_default_request_handler_with_custom_components(self) -> None:
        """Test DefaultRequestHandler initialization with custom components."""
        # Arrange
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        mock_queue_manager = AsyncMock(spec=QueueManager)
        mock_push_notifier = AsyncMock(spec=PushNotifier)
        mock_request_context_builder = AsyncMock(spec=RequestContextBuilder)

        # Act
        handler = DefaultRequestHandler(
            agent_executor=mock_agent_executor,
            task_store=mock_task_store,
            queue_manager=mock_queue_manager,
            push_notifier=mock_push_notifier,
            request_context_builder=mock_request_context_builder,
        )

        # Assert
        self.assertEqual(handler.agent_executor, mock_agent_executor)
        self.assertEqual(handler.task_store, mock_task_store)
        self.assertEqual(handler._queue_manager, mock_queue_manager)
        self.assertEqual(handler._push_notifier, mock_push_notifier)
        self.assertEqual(
            handler._request_context_builder, mock_request_context_builder
        )

    async def test_on_message_send_error_handling(self) -> None:
        """Test error handling in on_message_send when consuming raises ServerError."""
        # Arrange
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)

        # Let task exist
        mock_task = Task(**MINIMAL_TASK)
        mock_task_store.get.return_value = mock_task

        # Set up consume_and_break_on_interrupt to raise ServerError
        async def consume_raises_error(*args, **kwargs):
            raise ServerError(error=UnsupportedOperationError())

        with patch(
            'a2a.server.tasks.result_aggregator.ResultAggregator.consume_and_break_on_interrupt',
            side_effect=consume_raises_error,
        ):
            # Act
            request = SendMessageRequest(
                id='1',
                params=MessageSendParams(
                    message=Message(
                        **MESSAGE_PAYLOAD,
                        taskId=mock_task.id,
                        contextId=mock_task.contextId,
                    )
                ),
            )

            response = await handler.on_message_send(request)

            # Assert
            self.assertIsInstance(response.root, JSONRPCErrorResponse)
            self.assertEqual(response.root.error, UnsupportedOperationError())

    async def test_on_message_send_task_id_mismatch(self) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        mock_task = Task(**MINIMAL_TASK)
        mock_task_store.get.return_value = mock_task
        mock_agent_executor.execute.return_value = None

        async def streaming_coro():
            yield mock_task

        with patch(
            'a2a.server.request_handlers.default_request_handler.EventConsumer.consume_all',
            return_value=streaming_coro(),
        ):
            request = SendMessageRequest(
                id='1',
                params=MessageSendParams(message=Message(**MESSAGE_PAYLOAD)),
            )
            response = await handler.on_message_send(request)
            assert mock_agent_executor.execute.call_count == 1
            self.assertIsInstance(response.root, JSONRPCErrorResponse)
            self.assertIsInstance(response.root.error, InternalError)  # type: ignore

    async def test_on_message_stream_task_id_mismatch(self) -> None:
        mock_agent_executor = AsyncMock(spec=AgentExecutor)
        mock_task_store = AsyncMock(spec=TaskStore)
        request_handler = DefaultRequestHandler(
            mock_agent_executor, mock_task_store
        )

        self.mock_agent_card.capabilities = AgentCapabilities(streaming=True)
        handler = JSONRPCHandler(self.mock_agent_card, request_handler)
        events: list[Any] = [Task(**MINIMAL_TASK)]

        async def streaming_coro():
            for event in events:
                yield event

        with patch(
            'a2a.server.request_handlers.default_request_handler.EventConsumer.consume_all',
            return_value=streaming_coro(),
        ):
            mock_task_store.get.return_value = None
            mock_agent_executor.execute.return_value = None
            request = SendStreamingMessageRequest(
                id='1',
                params=MessageSendParams(message=Message(**MESSAGE_PAYLOAD)),
            )
            response = handler.on_message_send_stream(request)
            assert isinstance(response, AsyncGenerator)
            collected_events: list[Any] = []
            async for event in response:
                collected_events.append(event)
            assert len(collected_events) == 1
            self.assertIsInstance(
                collected_events[0].root, JSONRPCErrorResponse
            )
            self.assertIsInstance(collected_events[0].root.error, InternalError)
````

## File: src/a2a/types.py
````python
# generated by datamodel-codegen:
#   filename:  https://raw.githubusercontent.com/google-a2a/A2A/refs/heads/main/specification/json/a2a.json

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, RootModel


class A2A(RootModel[Any]):
    root: Any


class In(Enum):
    """
    The location of the API key. Valid values are "query", "header", or "cookie".
    """

    cookie = 'cookie'
    header = 'header'
    query = 'query'


class APIKeySecurityScheme(BaseModel):
    """
    API Key security scheme.
    """

    description: str | None = None
    """
    Description of this security scheme.
    """
    in_: In = Field(..., alias='in')
    """
    The location of the API key. Valid values are "query", "header", or "cookie".
    """
    name: str
    """
    The name of the header, query or cookie parameter to be used.
    """
    type: Literal['apiKey'] = 'apiKey'


class AgentCapabilities(BaseModel):
    """
    Defines optional capabilities supported by an agent.
    """

    pushNotifications: bool | None = None
    """
    true if the agent can notify updates to client.
    """
    stateTransitionHistory: bool | None = None
    """
    true if the agent exposes status change history for tasks.
    """
    streaming: bool | None = None
    """
    true if the agent supports SSE.
    """


class AgentProvider(BaseModel):
    """
    Represents the service provider of an agent.
    """

    organization: str
    """
    Agent provider's organization name.
    """
    url: str
    """
    Agent provider's URL.
    """


class AgentSkill(BaseModel):
    """
    Represents a unit of capability that an agent can perform.
    """

    description: str
    """
    Description of the skill - will be used by the client or a human
    as a hint to understand what the skill does.
    """
    examples: list[str] | None = None
    """
    The set of example scenarios that the skill can perform.
    Will be used by the client as a hint to understand how the skill can be used.
    """
    id: str
    """
    Unique identifier for the agent's skill.
    """
    inputModes: list[str] | None = None
    """
    The set of interaction modes that the skill supports
    (if different than the default).
    Supported mime types for input.
    """
    name: str
    """
    Human readable name of the skill.
    """
    outputModes: list[str] | None = None
    """
    Supported mime types for output.
    """
    tags: list[str]
    """
    Set of tagwords describing classes of capabilities for this specific skill.
    """


class AuthorizationCodeOAuthFlow(BaseModel):
    """
    Configuration details for a supported OAuth Flow
    """

    authorizationUrl: str
    """
    The authorization URL to be used for this flow. This MUST be in the form of a URL. The OAuth2
    standard requires the use of TLS
    """
    refreshUrl: str | None = None
    """
    The URL to be used for obtaining refresh tokens. This MUST be in the form of a URL. The OAuth2
    standard requires the use of TLS.
    """
    scopes: dict[str, str]
    """
    The available scopes for the OAuth2 security scheme. A map between the scope name and a short
    description for it. The map MAY be empty.
    """
    tokenUrl: str
    """
    The token URL to be used for this flow. This MUST be in the form of a URL. The OAuth2 standard
    requires the use of TLS.
    """


class ClientCredentialsOAuthFlow(BaseModel):
    """
    Configuration details for a supported OAuth Flow
    """

    refreshUrl: str | None = None
    """
    The URL to be used for obtaining refresh tokens. This MUST be in the form of a URL. The OAuth2
    standard requires the use of TLS.
    """
    scopes: dict[str, str]
    """
    The available scopes for the OAuth2 security scheme. A map between the scope name and a short
    description for it. The map MAY be empty.
    """
    tokenUrl: str
    """
    The token URL to be used for this flow. This MUST be in the form of a URL. The OAuth2 standard
    requires the use of TLS.
    """


class ContentTypeNotSupportedError(BaseModel):
    """
    A2A specific error indicating incompatible content types between request and agent capabilities.
    """

    code: Literal[-32005] = -32005
    """
    A Number that indicates the error type that occurred.
    """
    data: Any | None = None
    """
    A Primitive or Structured value that contains additional information about the error.
    This may be omitted.
    """
    message: str | None = 'Incompatible content types'
    """
    A String providing a short description of the error.
    """


class DataPart(BaseModel):
    """
    Represents a structured data segment within a message part.
    """

    data: dict[str, Any]
    """
    Structured data content
    """
    kind: Literal['data'] = 'data'
    """
    Part type - data for DataParts
    """
    metadata: dict[str, Any] | None = None
    """
    Optional metadata associated with the part.
    """


class FileBase(BaseModel):
    """
    Represents the base entity for FileParts
    """

    mimeType: str | None = None
    """
    Optional mimeType for the file
    """
    name: str | None = None
    """
    Optional name for the file
    """


class FileWithBytes(BaseModel):
    """
    Define the variant where 'bytes' is present and 'uri' is absent
    """

    bytes: str
    """
    base64 encoded content of the file
    """
    mimeType: str | None = None
    """
    Optional mimeType for the file
    """
    name: str | None = None
    """
    Optional name for the file
    """


class FileWithUri(BaseModel):
    """
    Define the variant where 'uri' is present and 'bytes' is absent
    """

    mimeType: str | None = None
    """
    Optional mimeType for the file
    """
    name: str | None = None
    """
    Optional name for the file
    """
    uri: str
    """
    URL for the File content
    """


class HTTPAuthSecurityScheme(BaseModel):
    """
    HTTP Authentication security scheme.
    """

    bearerFormat: str | None = None
    """
    A hint to the client to identify how the bearer token is formatted. Bearer tokens are usually
    generated by an authorization server, so this information is primarily for documentation
    purposes.
    """
    description: str | None = None
    """
    Description of this security scheme.
    """
    scheme: str
    """
    The name of the HTTP Authentication scheme to be used in the Authorization header as defined
    in RFC7235. The values used SHOULD be registered in the IANA Authentication Scheme registry.
    The value is case-insensitive, as defined in RFC7235.
    """
    type: Literal['http'] = 'http'


class ImplicitOAuthFlow(BaseModel):
    """
    Configuration details for a supported OAuth Flow
    """

    authorizationUrl: str
    """
    The authorization URL to be used for this flow. This MUST be in the form of a URL. The OAuth2
    standard requires the use of TLS
    """
    refreshUrl: str | None = None
    """
    The URL to be used for obtaining refresh tokens. This MUST be in the form of a URL. The OAuth2
    standard requires the use of TLS.
    """
    scopes: dict[str, str]
    """
    The available scopes for the OAuth2 security scheme. A map between the scope name and a short
    description for it. The map MAY be empty.
    """


class InternalError(BaseModel):
    """
    JSON-RPC error indicating an internal JSON-RPC error on the server.
    """

    code: Literal[-32603] = -32603
    """
    A Number that indicates the error type that occurred.
    """
    data: Any | None = None
    """
    A Primitive or Structured value that contains additional information about the error.
    This may be omitted.
    """
    message: str | None = 'Internal error'
    """
    A String providing a short description of the error.
    """


class InvalidAgentResponseError(BaseModel):
    """
    A2A specific error indicating agent returned invalid response for the current method
    """

    code: Literal[-32006] = -32006
    """
    A Number that indicates the error type that occurred.
    """
    data: Any | None = None
    """
    A Primitive or Structured value that contains additional information about the error.
    This may be omitted.
    """
    message: str | None = 'Invalid agent response'
    """
    A String providing a short description of the error.
    """


class InvalidParamsError(BaseModel):
    """
    JSON-RPC error indicating invalid method parameter(s).
    """

    code: Literal[-32602] = -32602
    """
    A Number that indicates the error type that occurred.
    """
    data: Any | None = None
    """
    A Primitive or Structured value that contains additional information about the error.
    This may be omitted.
    """
    message: str | None = 'Invalid parameters'
    """
    A String providing a short description of the error.
    """


class InvalidRequestError(BaseModel):
    """
    JSON-RPC error indicating the JSON sent is not a valid Request object.
    """

    code: Literal[-32600] = -32600
    """
    A Number that indicates the error type that occurred.
    """
    data: Any | None = None
    """
    A Primitive or Structured value that contains additional information about the error.
    This may be omitted.
    """
    message: str | None = 'Request payload validation error'
    """
    A String providing a short description of the error.
    """


class JSONParseError(BaseModel):
    """
    JSON-RPC error indicating invalid JSON was received by the server.
    """

    code: Literal[-32700] = -32700
    """
    A Number that indicates the error type that occurred.
    """
    data: Any | None = None
    """
    A Primitive or Structured value that contains additional information about the error.
    This may be omitted.
    """
    message: str | None = 'Invalid JSON payload'
    """
    A String providing a short description of the error.
    """


class JSONRPCError(BaseModel):
    """
    Represents a JSON-RPC 2.0 Error object.
    This is typically included in a JSONRPCErrorResponse when an error occurs.
    """

    code: int
    """
    A Number that indicates the error type that occurred.
    """
    data: Any | None = None
    """
    A Primitive or Structured value that contains additional information about the error.
    This may be omitted.
    """
    message: str
    """
    A String providing a short description of the error.
    """


class JSONRPCMessage(BaseModel):
    """
    Base interface for any JSON-RPC 2.0 request or response.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """


class JSONRPCRequest(BaseModel):
    """
    Represents a JSON-RPC 2.0 Request object.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """
    method: str
    """
    A String containing the name of the method to be invoked.
    """
    params: dict[str, Any] | None = None
    """
    A Structured value that holds the parameter values to be used during the invocation of the method.
    """


class JSONRPCResult(BaseModel):
    """
    Represents a JSON-RPC 2.0 Result object.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """
    result: Any
    """
    The result object on success
    """


class Role(Enum):
    """
    Message sender's role
    """

    agent = 'agent'
    user = 'user'


class MethodNotFoundError(BaseModel):
    """
    JSON-RPC error indicating the method does not exist or is not available.
    """

    code: Literal[-32601] = -32601
    """
    A Number that indicates the error type that occurred.
    """
    data: Any | None = None
    """
    A Primitive or Structured value that contains additional information about the error.
    This may be omitted.
    """
    message: str | None = 'Method not found'
    """
    A String providing a short description of the error.
    """


class OpenIdConnectSecurityScheme(BaseModel):
    """
    OpenID Connect security scheme configuration.
    """

    description: str | None = None
    """
    Description of this security scheme.
    """
    openIdConnectUrl: str
    """
    Well-known URL to discover the [[OpenID-Connect-Discovery]] provider metadata.
    """
    type: Literal['openIdConnect'] = 'openIdConnect'


class PartBase(BaseModel):
    """
    Base properties common to all message parts.
    """

    metadata: dict[str, Any] | None = None
    """
    Optional metadata associated with the part.
    """


class PasswordOAuthFlow(BaseModel):
    """
    Configuration details for a supported OAuth Flow
    """

    refreshUrl: str | None = None
    """
    The URL to be used for obtaining refresh tokens. This MUST be in the form of a URL. The OAuth2
    standard requires the use of TLS.
    """
    scopes: dict[str, str]
    """
    The available scopes for the OAuth2 security scheme. A map between the scope name and a short
    description for it. The map MAY be empty.
    """
    tokenUrl: str
    """
    The token URL to be used for this flow. This MUST be in the form of a URL. The OAuth2 standard
    requires the use of TLS.
    """


class PushNotificationAuthenticationInfo(BaseModel):
    """
    Defines authentication details for push notifications.
    """

    credentials: str | None = None
    """
    Optional credentials
    """
    schemes: list[str]
    """
    Supported authentication schemes - e.g. Basic, Bearer
    """


class PushNotificationConfig(BaseModel):
    """
    Configuration for setting up push notifications for task updates.
    """

    authentication: PushNotificationAuthenticationInfo | None = None
    token: str | None = None
    """
    Token unique to this task/session.
    """
    url: str
    """
    URL for sending the push notifications.
    """


class PushNotificationNotSupportedError(BaseModel):
    """
    A2A specific error indicating the agent does not support push notifications.
    """

    code: Literal[-32003] = -32003
    """
    A Number that indicates the error type that occurred.
    """
    data: Any | None = None
    """
    A Primitive or Structured value that contains additional information about the error.
    This may be omitted.
    """
    message: str | None = 'Push Notification is not supported'
    """
    A String providing a short description of the error.
    """


class SecuritySchemeBase(BaseModel):
    """
    Base properties shared by all security schemes.
    """

    description: str | None = None
    """
    Description of this security scheme.
    """


class TaskIdParams(BaseModel):
    """
    Parameters containing only a task ID, used for simple task operations.
    """

    id: str
    """
    Task id.
    """
    metadata: dict[str, Any] | None = None


class TaskNotCancelableError(BaseModel):
    """
    A2A specific error indicating the task is in a state where it cannot be canceled.
    """

    code: Literal[-32002] = -32002
    """
    A Number that indicates the error type that occurred.
    """
    data: Any | None = None
    """
    A Primitive or Structured value that contains additional information about the error.
    This may be omitted.
    """
    message: str | None = 'Task cannot be canceled'
    """
    A String providing a short description of the error.
    """


class TaskNotFoundError(BaseModel):
    """
    A2A specific error indicating the requested task ID was not found.
    """

    code: Literal[-32001] = -32001
    """
    A Number that indicates the error type that occurred.
    """
    data: Any | None = None
    """
    A Primitive or Structured value that contains additional information about the error.
    This may be omitted.
    """
    message: str | None = 'Task not found'
    """
    A String providing a short description of the error.
    """


class TaskPushNotificationConfig(BaseModel):
    """
    Parameters for setting or getting push notification configuration for a task
    """

    pushNotificationConfig: PushNotificationConfig
    """
    Push notification configuration.
    """
    taskId: str
    """
    Task id.
    """


class TaskQueryParams(BaseModel):
    """
    Parameters for querying a task, including optional history length.
    """

    historyLength: int | None = None
    """
    Number of recent messages to be retrieved.
    """
    id: str
    """
    Task id.
    """
    metadata: dict[str, Any] | None = None


class TaskResubscriptionRequest(BaseModel):
    """
    JSON-RPC request model for the 'tasks/resubscribe' method.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """
    method: Literal['tasks/resubscribe'] = 'tasks/resubscribe'
    """
    A String containing the name of the method to be invoked.
    """
    params: TaskIdParams
    """
    A Structured value that holds the parameter values to be used during the invocation of the method.
    """


class TaskState(Enum):
    """
    Represents the possible states of a Task.
    """

    submitted = 'submitted'
    working = 'working'
    input_required = 'input-required'
    completed = 'completed'
    canceled = 'canceled'
    failed = 'failed'
    rejected = 'rejected'
    auth_required = 'auth-required'
    unknown = 'unknown'


class TextPart(BaseModel):
    """
    Represents a text segment within parts.
    """

    kind: Literal['text'] = 'text'
    """
    Part type - text for TextParts
    """
    metadata: dict[str, Any] | None = None
    """
    Optional metadata associated with the part.
    """
    text: str
    """
    Text content
    """


class UnsupportedOperationError(BaseModel):
    """
    A2A specific error indicating the requested operation is not supported by the agent.
    """

    code: Literal[-32004] = -32004
    """
    A Number that indicates the error type that occurred.
    """
    data: Any | None = None
    """
    A Primitive or Structured value that contains additional information about the error.
    This may be omitted.
    """
    message: str | None = 'This operation is not supported'
    """
    A String providing a short description of the error.
    """


class A2AError(
    RootModel[
        JSONParseError
        | InvalidRequestError
        | MethodNotFoundError
        | InvalidParamsError
        | InternalError
        | TaskNotFoundError
        | TaskNotCancelableError
        | PushNotificationNotSupportedError
        | UnsupportedOperationError
        | ContentTypeNotSupportedError
        | InvalidAgentResponseError
    ]
):
    root: (
        JSONParseError
        | InvalidRequestError
        | MethodNotFoundError
        | InvalidParamsError
        | InternalError
        | TaskNotFoundError
        | TaskNotCancelableError
        | PushNotificationNotSupportedError
        | UnsupportedOperationError
        | ContentTypeNotSupportedError
        | InvalidAgentResponseError
    )


class CancelTaskRequest(BaseModel):
    """
    JSON-RPC request model for the 'tasks/cancel' method.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """
    method: Literal['tasks/cancel'] = 'tasks/cancel'
    """
    A String containing the name of the method to be invoked.
    """
    params: TaskIdParams
    """
    A Structured value that holds the parameter values to be used during the invocation of the method.
    """


class FilePart(BaseModel):
    """
    Represents a File segment within parts.
    """

    file: FileWithBytes | FileWithUri
    """
    File content either as url or bytes
    """
    kind: Literal['file'] = 'file'
    """
    Part type - file for FileParts
    """
    metadata: dict[str, Any] | None = None
    """
    Optional metadata associated with the part.
    """


class GetTaskPushNotificationConfigRequest(BaseModel):
    """
    JSON-RPC request model for the 'tasks/pushNotificationConfig/get' method.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """
    method: Literal['tasks/pushNotificationConfig/get'] = (
        'tasks/pushNotificationConfig/get'
    )
    """
    A String containing the name of the method to be invoked.
    """
    params: TaskIdParams
    """
    A Structured value that holds the parameter values to be used during the invocation of the method.
    """


class GetTaskPushNotificationConfigSuccessResponse(BaseModel):
    """
    JSON-RPC success response model for the 'tasks/pushNotificationConfig/get' method.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """
    result: TaskPushNotificationConfig
    """
    The result object on success.
    """


class GetTaskRequest(BaseModel):
    """
    JSON-RPC request model for the 'tasks/get' method.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """
    method: Literal['tasks/get'] = 'tasks/get'
    """
    A String containing the name of the method to be invoked.
    """
    params: TaskQueryParams
    """
    A Structured value that holds the parameter values to be used during the invocation of the method.
    """


class JSONRPCErrorResponse(BaseModel):
    """
    Represents a JSON-RPC 2.0 Error Response object.
    """

    error: (
        JSONRPCError
        | JSONParseError
        | InvalidRequestError
        | MethodNotFoundError
        | InvalidParamsError
        | InternalError
        | TaskNotFoundError
        | TaskNotCancelableError
        | PushNotificationNotSupportedError
        | UnsupportedOperationError
        | ContentTypeNotSupportedError
        | InvalidAgentResponseError
    )
    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """


class MessageSendConfiguration(BaseModel):
    """
    Configuration for the send message request.
    """

    acceptedOutputModes: list[str]
    """
    Accepted output modalities by the client.
    """
    blocking: bool | None = None
    """
    If the server should treat the client as a blocking request.
    """
    historyLength: int | None = None
    """
    Number of recent messages to be retrieved.
    """
    pushNotificationConfig: PushNotificationConfig | None = None
    """
    Where the server should send notifications when disconnected.
    """


class OAuthFlows(BaseModel):
    """
    Allows configuration of the supported OAuth Flows
    """

    authorizationCode: AuthorizationCodeOAuthFlow | None = None
    """
    Configuration for the OAuth Authorization Code flow. Previously called accessCode in OpenAPI 2.0.
    """
    clientCredentials: ClientCredentialsOAuthFlow | None = None
    """
    Configuration for the OAuth Client Credentials flow. Previously called application in OpenAPI 2.0
    """
    implicit: ImplicitOAuthFlow | None = None
    """
    Configuration for the OAuth Implicit flow
    """
    password: PasswordOAuthFlow | None = None
    """
    Configuration for the OAuth Resource Owner Password flow
    """


class Part(RootModel[TextPart | FilePart | DataPart]):
    root: TextPart | FilePart | DataPart
    """
    Represents a part of a message, which can be text, a file, or structured data.
    """


class SetTaskPushNotificationConfigRequest(BaseModel):
    """
    JSON-RPC request model for the 'tasks/pushNotificationConfig/set' method.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """
    method: Literal['tasks/pushNotificationConfig/set'] = (
        'tasks/pushNotificationConfig/set'
    )
    """
    A String containing the name of the method to be invoked.
    """
    params: TaskPushNotificationConfig
    """
    A Structured value that holds the parameter values to be used during the invocation of the method.
    """


class SetTaskPushNotificationConfigSuccessResponse(BaseModel):
    """
    JSON-RPC success response model for the 'tasks/pushNotificationConfig/set' method.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """
    result: TaskPushNotificationConfig
    """
    The result object on success.
    """


class Artifact(BaseModel):
    """
    Represents an artifact generated for a task.
    """

    artifactId: str
    """
    Unique identifier for the artifact.
    """
    description: str | None = None
    """
    Optional description for the artifact.
    """
    metadata: dict[str, Any] | None = None
    """
    Extension metadata.
    """
    name: str | None = None
    """
    Optional name for the artifact.
    """
    parts: list[Part]
    """
    Artifact parts.
    """


class GetTaskPushNotificationConfigResponse(
    RootModel[
        JSONRPCErrorResponse | GetTaskPushNotificationConfigSuccessResponse
    ]
):
    root: JSONRPCErrorResponse | GetTaskPushNotificationConfigSuccessResponse
    """
    JSON-RPC response for the 'tasks/pushNotificationConfig/set' method.
    """


class Message(BaseModel):
    """
    Represents a single message exchanged between user and agent.
    """

    contextId: str | None = None
    """
    The context the message is associated with
    """
    kind: Literal['message'] = 'message'
    """
    Event type
    """
    messageId: str
    """
    Identifier created by the message creator
    """
    metadata: dict[str, Any] | None = None
    """
    Extension metadata.
    """
    parts: list[Part]
    """
    Message content
    """
    referenceTaskIds: list[str] | None = None
    """
    List of tasks referenced as context by this message.
    """
    role: Role
    """
    Message sender's role
    """
    taskId: str | None = None
    """
    Identifier of task the message is related to
    """


class MessageSendParams(BaseModel):
    """
    Sent by the client to the agent as a request. May create, continue or restart a task.
    """

    configuration: MessageSendConfiguration | None = None
    """
    Send message configuration.
    """
    message: Message
    """
    The message being sent to the server.
    """
    metadata: dict[str, Any] | None = None
    """
    Extension metadata.
    """


class OAuth2SecurityScheme(BaseModel):
    """
    OAuth2.0 security scheme configuration.
    """

    description: str | None = None
    """
    Description of this security scheme.
    """
    flows: OAuthFlows
    """
    An object containing configuration information for the flow types supported.
    """
    type: Literal['oauth2'] = 'oauth2'


class SecurityScheme(
    RootModel[
        APIKeySecurityScheme
        | HTTPAuthSecurityScheme
        | OAuth2SecurityScheme
        | OpenIdConnectSecurityScheme
    ]
):
    root: (
        APIKeySecurityScheme
        | HTTPAuthSecurityScheme
        | OAuth2SecurityScheme
        | OpenIdConnectSecurityScheme
    )
    """
    Mirrors the OpenAPI Security Scheme Object
    (https://swagger.io/specification/#security-scheme-object)
    """


class SendMessageRequest(BaseModel):
    """
    JSON-RPC request model for the 'message/send' method.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """
    method: Literal['message/send'] = 'message/send'
    """
    A String containing the name of the method to be invoked.
    """
    params: MessageSendParams
    """
    A Structured value that holds the parameter values to be used during the invocation of the method.
    """


class SendStreamingMessageRequest(BaseModel):
    """
    JSON-RPC request model for the 'message/stream' method.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """
    method: Literal['message/stream'] = 'message/stream'
    """
    A String containing the name of the method to be invoked.
    """
    params: MessageSendParams
    """
    A Structured value that holds the parameter values to be used during the invocation of the method.
    """


class SetTaskPushNotificationConfigResponse(
    RootModel[
        JSONRPCErrorResponse | SetTaskPushNotificationConfigSuccessResponse
    ]
):
    root: JSONRPCErrorResponse | SetTaskPushNotificationConfigSuccessResponse
    """
    JSON-RPC response for the 'tasks/pushNotificationConfig/set' method.
    """


class TaskArtifactUpdateEvent(BaseModel):
    """
    Sent by server during sendStream or subscribe requests
    """

    append: bool | None = None
    """
    Indicates if this artifact appends to a previous one
    """
    artifact: Artifact
    """
    Generated artifact
    """
    contextId: str
    """
    The context the task is associated with
    """
    kind: Literal['artifact-update'] = 'artifact-update'
    """
    Event type
    """
    lastChunk: bool | None = None
    """
    Indicates if this is the last chunk of the artifact
    """
    metadata: dict[str, Any] | None = None
    """
    Extension metadata.
    """
    taskId: str
    """
    Task id
    """


class TaskStatus(BaseModel):
    """
    TaskState and accompanying message.
    """

    message: Message | None = None
    """
    Additional status updates for client
    """
    state: TaskState
    timestamp: str | None = None
    """
    ISO 8601 datetime string when the status was recorded.
    """


class TaskStatusUpdateEvent(BaseModel):
    """
    Sent by server during sendStream or subscribe requests
    """

    contextId: str
    """
    The context the task is associated with
    """
    final: bool
    """
    Indicates the end of the event stream
    """
    kind: Literal['status-update'] = 'status-update'
    """
    Event type
    """
    metadata: dict[str, Any] | None = None
    """
    Extension metadata.
    """
    status: TaskStatus
    """
    Current status of the task
    """
    taskId: str
    """
    Task id
    """


class A2ARequest(
    RootModel[
        SendMessageRequest
        | SendStreamingMessageRequest
        | GetTaskRequest
        | CancelTaskRequest
        | SetTaskPushNotificationConfigRequest
        | GetTaskPushNotificationConfigRequest
        | TaskResubscriptionRequest
    ]
):
    root: (
        SendMessageRequest
        | SendStreamingMessageRequest
        | GetTaskRequest
        | CancelTaskRequest
        | SetTaskPushNotificationConfigRequest
        | GetTaskPushNotificationConfigRequest
        | TaskResubscriptionRequest
    )
    """
    A2A supported request types
    """


class AgentCard(BaseModel):
    """
    An AgentCard conveys key information:
    - Overall details (version, name, description, uses)
    - Skills: A set of capabilities the agent can perform
    - Default modalities/content types supported by the agent.
    - Authentication requirements
    """

    capabilities: AgentCapabilities
    """
    Optional capabilities supported by the agent.
    """
    defaultInputModes: list[str]
    """
    The set of interaction modes that the agent supports across all skills. This can be overridden per-skill.
    Supported mime types for input.
    """
    defaultOutputModes: list[str]
    """
    Supported mime types for output.
    """
    description: str
    """
    A human-readable description of the agent. Used to assist users and
    other agents in understanding what the agent can do.
    """
    documentationUrl: str | None = None
    """
    A URL to documentation for the agent.
    """
    name: str
    """
    Human readable name of the agent.
    """
    provider: AgentProvider | None = None
    """
    The service provider of the agent
    """
    security: list[dict[str, list[str]]] | None = None
    """
    Security requirements for contacting the agent.
    """
    securitySchemes: dict[str, SecurityScheme] | None = None
    """
    Security scheme details used for authenticating with this agent.
    """
    skills: list[AgentSkill]
    """
    Skills are a unit of capability that an agent can perform.
    """
    supportsAuthenticatedExtendedCard: bool | None = None
    """
    true if the agent supports providing an extended agent card when the user is authenticated.
    Defaults to false if not specified.
    """
    url: str
    """
    A URL to the address the agent is hosted at.
    """
    version: str
    """
    The version of the agent - format is up to the provider.
    """
    supportsAuthenticatedExtendedCard: bool | None = Field(default=None)
    """
    Optional field indicating there is an extended card available post authentication at the /agent/authenticatedExtendedCard endpoint.
    """


class Task(BaseModel):
    artifacts: list[Artifact] | None = None
    """
    Collection of artifacts created by the agent.
    """
    contextId: str
    """
    Server-generated id for contextual alignment across interactions
    """
    history: list[Message] | None = None
    id: str
    """
    Unique identifier for the task
    """
    kind: Literal['task'] = 'task'
    """
    Event type
    """
    metadata: dict[str, Any] | None = None
    """
    Extension metadata.
    """
    status: TaskStatus
    """
    Current status of the task
    """


class CancelTaskSuccessResponse(BaseModel):
    """
    JSON-RPC success response model for the 'tasks/cancel' method.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """
    result: Task
    """
    The result object on success.
    """


class GetTaskSuccessResponse(BaseModel):
    """
    JSON-RPC success response for the 'tasks/get' method.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """
    result: Task
    """
    The result object on success.
    """


class SendMessageSuccessResponse(BaseModel):
    """
    JSON-RPC success response model for the 'message/send' method.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """
    result: Task | Message
    """
    The result object on success
    """


class SendStreamingMessageSuccessResponse(BaseModel):
    """
    JSON-RPC success response model for the 'message/stream' method.
    """

    id: str | int | None = None
    """
    An identifier established by the Client that MUST contain a String, Number.
    Numbers SHOULD NOT contain fractional parts.
    """
    jsonrpc: Literal['2.0'] = '2.0'
    """
    Specifies the version of the JSON-RPC protocol. MUST be exactly "2.0".
    """
    result: Task | Message | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
    """
    The result object on success
    """


class CancelTaskResponse(
    RootModel[JSONRPCErrorResponse | CancelTaskSuccessResponse]
):
    root: JSONRPCErrorResponse | CancelTaskSuccessResponse
    """
    JSON-RPC response for the 'tasks/cancel' method.
    """


class GetTaskResponse(RootModel[JSONRPCErrorResponse | GetTaskSuccessResponse]):
    root: JSONRPCErrorResponse | GetTaskSuccessResponse
    """
    JSON-RPC success response for the 'tasks/get' method.
    """


class JSONRPCResponse(
    RootModel[
        JSONRPCErrorResponse
        | SendMessageSuccessResponse
        | SendStreamingMessageSuccessResponse
        | GetTaskSuccessResponse
        | CancelTaskSuccessResponse
        | SetTaskPushNotificationConfigSuccessResponse
        | GetTaskPushNotificationConfigSuccessResponse
    ]
):
    root: (
        JSONRPCErrorResponse
        | SendMessageSuccessResponse
        | SendStreamingMessageSuccessResponse
        | GetTaskSuccessResponse
        | CancelTaskSuccessResponse
        | SetTaskPushNotificationConfigSuccessResponse
        | GetTaskPushNotificationConfigSuccessResponse
    )
    """
    Represents a JSON-RPC 2.0 Response object.
    """


class SendMessageResponse(
    RootModel[JSONRPCErrorResponse | SendMessageSuccessResponse]
):
    root: JSONRPCErrorResponse | SendMessageSuccessResponse
    """
    JSON-RPC response model for the 'message/send' method.
    """


class SendStreamingMessageResponse(
    RootModel[JSONRPCErrorResponse | SendStreamingMessageSuccessResponse]
):
    root: JSONRPCErrorResponse | SendStreamingMessageSuccessResponse
    """
    JSON-RPC response model for the 'message/stream' method.
    """
````

## File: src/a2a/server/request_handlers/default_request_handler.py
````python
import asyncio
import logging

from collections.abc import AsyncGenerator
from typing import cast

from a2a.server.agent_execution import (
    AgentExecutor,
    RequestContext,
    RequestContextBuilder,
    SimpleRequestContextBuilder,
)
from a2a.server.context import ServerCallContext
from a2a.server.events import (
    Event,
    EventConsumer,
    EventQueue,
    InMemoryQueueManager,
    QueueManager,
)
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.tasks import (
    PushNotifier,
    ResultAggregator,
    TaskManager,
    TaskStore,
)
from a2a.types import (
    InternalError,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    PushNotificationConfig,
    Task,
    TaskIdParams,
    TaskNotFoundError,
    TaskPushNotificationConfig,
    TaskQueryParams,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from a2a.utils.telemetry import SpanKind, trace_class


logger = logging.getLogger(__name__)


@trace_class(kind=SpanKind.SERVER)
class DefaultRequestHandler(RequestHandler):
    """Default request handler for all incoming requests.

    This handler provides default implementations for all A2A JSON-RPC methods,
    coordinating between the `AgentExecutor`, `TaskStore`, `QueueManager`,
    and optional `PushNotifier`.
    """

    _running_agents: dict[str, asyncio.Task]

    def __init__(
        self,
        agent_executor: AgentExecutor,
        task_store: TaskStore,
        queue_manager: QueueManager | None = None,
        push_notifier: PushNotifier | None = None,
        request_context_builder: RequestContextBuilder | None = None,
    ) -> None:
        """Initializes the DefaultRequestHandler.

        Args:
            agent_executor: The `AgentExecutor` instance to run agent logic.
            task_store: The `TaskStore` instance to manage task persistence.
            queue_manager: The `QueueManager` instance to manage event queues. Defaults to `InMemoryQueueManager`.
            push_notifier: The `PushNotifier` instance for sending push notifications. Defaults to None.
            request_context_builder: The `RequestContextBuilder` instance used
              to build request contexts. Defaults to `SimpleRequestContextBuilder`.
        """
        self.agent_executor = agent_executor
        self.task_store = task_store
        self._queue_manager = queue_manager or InMemoryQueueManager()
        self._push_notifier = push_notifier
        self._request_context_builder = (
            request_context_builder
            or SimpleRequestContextBuilder(
                should_populate_referred_tasks=False, task_store=self.task_store
            )
        )
        # TODO: Likely want an interface for managing this, like AgentExecutionManager.
        self._running_agents = {}
        self._running_agents_lock = asyncio.Lock()

    async def on_get_task(
        self,
        params: TaskQueryParams,
        context: ServerCallContext | None = None,
    ) -> Task | None:
        """Default handler for 'tasks/get'."""
        task: Task | None = await self.task_store.get(params.id)
        if not task:
            raise ServerError(error=TaskNotFoundError())
        return task

    async def on_cancel_task(
        self, params: TaskIdParams, context: ServerCallContext | None = None
    ) -> Task | None:
        """Default handler for 'tasks/cancel'.

        Attempts to cancel the task managed by the `AgentExecutor`.
        """
        task: Task | None = await self.task_store.get(params.id)
        if not task:
            raise ServerError(error=TaskNotFoundError())

        task_manager = TaskManager(
            task_id=task.id,
            context_id=task.contextId,
            task_store=self.task_store,
            initial_message=None,
        )
        result_aggregator = ResultAggregator(task_manager)

        queue = await self._queue_manager.tap(task.id)
        if not queue:
            queue = EventQueue()

        await self.agent_executor.cancel(
            RequestContext(
                None,
                task_id=task.id,
                context_id=task.contextId,
                task=task,
            ),
            queue,
        )
        # Cancel the ongoing task, if one exists.
        if producer_task := self._running_agents.get(task.id):
            producer_task.cancel()

        consumer = EventConsumer(queue)
        result = await result_aggregator.consume_all(consumer)
        if isinstance(result, Task):
            return result

        raise ServerError(
            error=InternalError(
                message='Agent did not return valid response for cancel'
            )
        )

    async def _run_event_stream(
        self, request: RequestContext, queue: EventQueue
    ) -> None:
        """Runs the agent's `execute` method and closes the queue afterwards.

        Args:
            request: The request context for the agent.
            queue: The event queue for the agent to publish to.
        """
        await self.agent_executor.execute(request, queue)
        await queue.close()

    async def on_message_send(
        self,
        params: MessageSendParams,
        context: ServerCallContext | None = None,
    ) -> Message | Task:
        """Default handler for 'message/send' interface (non-streaming).

        Starts the agent execution for the message and waits for the final
        result (Task or Message).
        """
        task_manager = TaskManager(
            task_id=params.message.taskId,
            context_id=params.message.contextId,
            task_store=self.task_store,
            initial_message=params.message,
        )
        task: Task | None = await task_manager.get_task()
        if task:
            task = task_manager.update_with_message(params.message, task)
            if self.should_add_push_info(params):
                assert isinstance(self._push_notifier, PushNotifier)
                assert isinstance(
                    params.configuration, MessageSendConfiguration
                )
                assert isinstance(
                    params.configuration.pushNotificationConfig,
                    PushNotificationConfig,
                )
                await self._push_notifier.set_info(
                    task.id, params.configuration.pushNotificationConfig
                )
        request_context = await self._request_context_builder.build(
            params=params,
            task_id=task.id if task else None,
            context_id=params.message.contextId,
            task=task,
            context=context,
        )

        task_id = cast(str, request_context.task_id)
        # Always assign a task ID. We may not actually upgrade to a task, but
        # dictating the task ID at this layer is useful for tracking running
        # agents.
        queue = await self._queue_manager.create_or_tap(task_id)
        result_aggregator = ResultAggregator(task_manager)
        # TODO: to manage the non-blocking flows.
        producer_task = asyncio.create_task(
            self._run_event_stream(
                request_context,
                queue,
            )
        )
        await self._register_producer(task_id, producer_task)

        consumer = EventConsumer(queue)
        producer_task.add_done_callback(consumer.agent_task_callback)

        interrupted = False
        try:
            (
                result,
                interrupted,
            ) = await result_aggregator.consume_and_break_on_interrupt(consumer)
            if not result:
                raise ServerError(error=InternalError())

            if isinstance(result, Task) and task_id != result.id:
                logger.error(
                    f'Agent generated task_id={result.id} does not match the RequestContext task_id={task_id}.'
                )
                raise ServerError(
                    InternalError(message='Task ID mismatch in agent response')
                )

        finally:
            if interrupted:
                # TODO: Track this disconnected cleanup task.
                asyncio.create_task(
                    self._cleanup_producer(producer_task, task_id)
                )
            else:
                await self._cleanup_producer(producer_task, task_id)

        return result

    async def on_message_send_stream(
        self,
        params: MessageSendParams,
        context: ServerCallContext | None = None,
    ) -> AsyncGenerator[Event]:
        """Default handler for 'message/stream' (streaming).

        Starts the agent execution and yields events as they are produced
        by the agent.
        """
        task_manager = TaskManager(
            task_id=params.message.taskId,
            context_id=params.message.contextId,
            task_store=self.task_store,
            initial_message=params.message,
        )
        task: Task | None = await task_manager.get_task()

        if task:
            task = task_manager.update_with_message(params.message, task)

            if self.should_add_push_info(params):
                assert isinstance(self._push_notifier, PushNotifier)
                assert isinstance(
                    params.configuration, MessageSendConfiguration
                )
                assert isinstance(
                    params.configuration.pushNotificationConfig,
                    PushNotificationConfig,
                )
                await self._push_notifier.set_info(
                    task.id, params.configuration.pushNotificationConfig
                )
        else:
            queue = EventQueue()
        result_aggregator = ResultAggregator(task_manager)
        request_context = await self._request_context_builder.build(
            params=params,
            task_id=task.id if task else None,
            context_id=params.message.contextId,
            task=task,
            context=context,
        )

        task_id = cast(str, request_context.task_id)
        queue = await self._queue_manager.create_or_tap(task_id)
        producer_task = asyncio.create_task(
            self._run_event_stream(
                request_context,
                queue,
            )
        )
        await self._register_producer(task_id, producer_task)

        try:
            consumer = EventConsumer(queue)
            producer_task.add_done_callback(consumer.agent_task_callback)
            async for event in result_aggregator.consume_and_emit(consumer):
                if isinstance(event, Task):
                    if task_id != event.id:
                        logger.error(
                            f'Agent generated task_id={event.id} does not match the RequestContext task_id={task_id}.'
                        )
                        raise ServerError(
                            InternalError(
                                message='Task ID mismatch in agent response'
                            )
                        )

                    if (
                        self._push_notifier
                        and params.configuration
                        and params.configuration.pushNotificationConfig
                    ):
                        await self._push_notifier.set_info(
                            task_id,
                            params.configuration.pushNotificationConfig,
                        )

                if self._push_notifier and task_id:
                    latest_task = await result_aggregator.current_result
                    if isinstance(latest_task, Task):
                        await self._push_notifier.send_notification(latest_task)
                yield event
        finally:
            await self._cleanup_producer(producer_task, task_id)

    async def _register_producer(
        self, task_id: str, producer_task: asyncio.Task
    ) -> None:
        """Registers the agent execution task with the handler."""
        async with self._running_agents_lock:
            self._running_agents[task_id] = producer_task

    async def _cleanup_producer(
        self,
        producer_task: asyncio.Task,
        task_id: str,
    ) -> None:
        """Cleans up the agent execution task and queue manager entry."""
        await producer_task
        await self._queue_manager.close(task_id)
        async with self._running_agents_lock:
            self._running_agents.pop(task_id, None)

    async def on_set_task_push_notification_config(
        self,
        params: TaskPushNotificationConfig,
        context: ServerCallContext | None = None,
    ) -> TaskPushNotificationConfig:
        """Default handler for 'tasks/pushNotificationConfig/set'.

        Requires a `PushNotifier` to be configured.
        """
        if not self._push_notifier:
            raise ServerError(error=UnsupportedOperationError())

        task: Task | None = await self.task_store.get(params.taskId)
        if not task:
            raise ServerError(error=TaskNotFoundError())

        await self._push_notifier.set_info(
            params.taskId,
            params.pushNotificationConfig,
        )

        return params

    async def on_get_task_push_notification_config(
        self,
        params: TaskIdParams,
        context: ServerCallContext | None = None,
    ) -> TaskPushNotificationConfig:
        """Default handler for 'tasks/pushNotificationConfig/get'.

        Requires a `PushNotifier` to be configured.
        """
        if not self._push_notifier:
            raise ServerError(error=UnsupportedOperationError())

        task: Task | None = await self.task_store.get(params.id)
        if not task:
            raise ServerError(error=TaskNotFoundError())

        push_notification_config = await self._push_notifier.get_info(params.id)
        if not push_notification_config:
            raise ServerError(error=InternalError())

        return TaskPushNotificationConfig(
            taskId=params.id, pushNotificationConfig=push_notification_config
        )

    async def on_resubscribe_to_task(
        self,
        params: TaskIdParams,
        context: ServerCallContext | None = None,
    ) -> AsyncGenerator[Event]:
        """Default handler for 'tasks/resubscribe'.

        Allows a client to re-attach to a running streaming task's event stream.
        Requires the task and its queue to still be active.
        """
        task: Task | None = await self.task_store.get(params.id)
        if not task:
            raise ServerError(error=TaskNotFoundError())

        task_manager = TaskManager(
            task_id=task.id,
            context_id=task.contextId,
            task_store=self.task_store,
            initial_message=None,
        )

        result_aggregator = ResultAggregator(task_manager)

        queue = await self._queue_manager.tap(task.id)
        if not queue:
            raise ServerError(error=TaskNotFoundError())

        consumer = EventConsumer(queue)
        async for event in result_aggregator.consume_and_emit(consumer):
            yield event

    def should_add_push_info(self, params: MessageSendParams) -> bool:
        return bool(
            self._push_notifier
            and params.configuration
            and params.configuration.pushNotificationConfig
        )
````
