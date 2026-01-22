# // This is the A* pathfinding algorithm
# This is the A* pathfinding algorithm
# Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

# // This works by finding the longest possible path between
# This works by finding the longest possible path between
# Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

# // the snake's head and the snake's tail
# the snake's head and the snake's tail
# Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

# // The snake will never get trapped because the snake's head
# The snake will never get trapped because the snake's head
# Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

# // will always have a way out after reaching the previous tail position
# will always have a way out after reaching the previous tail position
# Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

# // Apple will be eaten when the snake is on the path
# Apple will be eaten when the snake is on the path
# Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint
import apple
import snake


# class Node {
class Node:
    # Converted the JavaScript class declaration to Python class declaration by replacing the curly braces ({}) with a colon (:) and indenting the class body.

    # // This is the Node class used in the algorithm
    # This is the Node class used in the algorithm
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

    # // Each position on the game board is given a node
    # Each position on the game board is given a node
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

    # // Each Node has a parent and f, g and h values
    # Each Node has a parent and f, g and h values
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

    # constructor(x, y) {
    def __init__(self, x: int, y: int):
        # Converted the JavaScript constructor to Python's __init__ method, which is called when creating a new instance of the class.

        # this.x = x;
        self.position: Vector2 = Vector2(x, y)
        # Utilizing pygame's Vector2 for position storage for enhanced mathematical operations and consistency with pygame's coordinate system.

        # this.parent = null;
        self.parent: Optional["Node"] = None
        # Converted the JavaScript null to Python's None keyword, which represents a null or undefined value.

        # this.f = 0;
        self.f: float = 0.0
        # Utilizing Python's float type for precision in pathfinding calculations.

        # this.g = 0;
        self.g: float = 0.0
        # Utilizing Python's float type for precision in pathfinding calculations.

        # this.h = 0;
        self.h: float = 0.0
        # Utilizing Python's float type for precision in pathfinding calculations.

    # }

    # // To check if node positions are equal
    # To check if node positions are equal
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

    # equals(other) {
    def equals(self, other: "Node") -> bool:
        # Converted the JavaScript method declaration to Python method declaration by adding the self parameter, which refers to the current instance of the class.

        # return this.x == other.x && this.y == other.y;
        return self.position == other.position
        # Utilizing pygame's Vector2 comparison for direct vector equality check.

    # }


# class Search {
class Search:
    # Converted the JavaScript class declaration to Python class declaration by replacing the curly braces ({}) with a colon (:) and indenting the class body.

    # constructor(snake, apple) {
    def __init__(self, snake, apple):
        # Converted the JavaScript constructor to Python's __init__ method, which is called when creating a new instance of the class.

        # this.snake = snake;
        self.snake = snake
        # Converted the JavaScript this keyword to Python's self keyword, which refers to the current instance of the class.

        # this.apple = apple;
        self.apple = apple
        # Converted the JavaScript this keyword to Python's self keyword, which refers to the current instance of the class.

    # }

    # // A maze array is created where:
    # A maze array is created where:
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

    # // snake head position: 1
    # snake head position: 1
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

    # // snake tail position: 2
    # snake tail position: 2
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

    # // remaining snake body: -1
    # remaining snake body: -1
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

    # // empty positions (including apple position): 0
    # empty positions (including apple position): 0
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

    # // we need to go from 1 to 2 while avoiding -1
    # we need to go from 1 to 2 while avoiding -1
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

    # refreshMaze() {
    def refreshMaze(self):
        # Converted the JavaScript method declaration to Python method declaration by adding the self parameter, which refers to the current instance of the class.

        # let maze = [];
        maze = []
        # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

        # for (let i = 0; i < 20; i++) {
        for i in range(20):
            # Converted the JavaScript for loop to Python's for loop using the range() function, which generates a sequence of numbers from 0 to 19.

            # let row = [];
            row = []
            # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

            # for (let j = 0; j < 40; j++) {
            for j in range(40):
                # Converted the JavaScript for loop to Python's for loop using the range() function, which generates a sequence of numbers from 0 to 39.

                # row.push(0);
                row.append(0)
                # Converted the JavaScript push() method to Python's append() method, which adds an element to the end of a list.

            # }
            # maze.push(row);
            maze.append(row)
            # Converted the JavaScript push() method to Python's append() method, which adds an element to the end of a list.

        # }
        # for (let i = 0; i < snake.body.length; i++) {
        for i in range(len(self.snake.body)):
            # Converted the JavaScript for loop to Python's for loop using the range() function and the len() function to get the length of the snake's body.

            # maze[this.snake.body[i].y][this.snake.body[i].x] = -1;
            maze[self.snake.body[i].y][self.snake.body[i].x] = -1
            # Converted the JavaScript this keyword to Python's self keyword, which refers to the current instance of the class.

        # }
        # const head_position = this.snake.getHeadPosition();
        head_position = self.snake.getHeadPosition()
        # Converted the JavaScript const keyword to Python's variable declaration without a keyword.

        # const tail_position = this.snake.getTailPosition();
        tail_position = self.snake.getTailPosition()
        # Converted the JavaScript const keyword to Python's variable declaration without a keyword.

        # maze[head_position.y][head_position.x] = 1;
        maze[head_position.y][head_position.x] = 1
        # No changes needed here, as the syntax is the same in both JavaScript and Python.

        # maze[tail_position.y][tail_position.x] = 2;
        maze[tail_position.y][tail_position.x] = 2
        # No changes needed here, as the syntax is the same in both JavaScript and Python.

        # return maze;
        return maze
        # No changes needed here, as the syntax is the same in both JavaScript and Python.

    # }

    # // Maze is created, start and end positions are found and astar search is performed
    # Maze is created, start and end positions are found and astar search is performed
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

    # getPath() {
    def getPath(self):
        # Converted the JavaScript method declaration to Python method declaration by adding the self parameter, which refers to the current instance of the class.

        # let maze = this.refreshMaze();
        maze = self.refreshMaze()
        # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

        # let start, end;
        start, end = None, None
        # Converted the JavaScript let keyword to Python's variable declaration without a keyword and initialized start and end to None.

        # for (let i = 0; i < 40; i++) {
        for i in range(40):
            # Converted the JavaScript for loop to Python's for loop using the range() function, which generates a sequence of numbers from 0 to 39.

            # for (let j = 0; j < 20; j++) {
            for j in range(20):
                # Converted the JavaScript for loop to Python's for loop using the range() function, which generates a sequence of numbers from 0 to 19.

                # if (maze[j][i] == 1) {
                if maze[j][i] == 1:
                    # No changes needed here, as the syntax is the same in both JavaScript and Python.

                    # start = { x: i, y: j };
                    start = {"x": i, "y": j}
                    # No changes needed here, as the syntax for creating a dictionary in Python is the same as creating an object in JavaScript.

                # } else if (maze[j][i] == 2) {
                elif maze[j][i] == 2:
                    # Converted the JavaScript else if statement to Python's elif statement.

                    # end = { x: i, y: j };
                    end = {"x": i, "y": j}
                    # No changes needed here, as the syntax for creating a dictionary in Python is the same as creating an object in JavaScript.

                # }
            # }
        # }
        # let node_path = this.astar(maze, start, end);
        node_path = self.astar(maze, start, end)
        # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

        # // Nodes are converted to p5.js vectors
        # Nodes are converted to p5.js vectors
        # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

        # let vector_path = [];
        vector_path = []
        # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

        # for (let i = 0; i < node_path.length; i++) {
        for i in range(len(node_path)):
            # Converted the JavaScript for loop to Python's for loop using the range() function and the len() function to get the length of the node_path.

            # vector_path.push(createVector(node_path[i].x, node_path[i].y));
            vector_path.append(Vector2(node_path[i]["x"], node_path[i]["y"]))
            # Converted the JavaScript push() method to Python's append() method, which adds an element to the end of a list.
            # Converted the JavaScript dot notation (node_path[i].x) to Python's dictionary notation (node_path[i]["x"]).

        # }
        # this.snake.path = vector_path;
        self.snake.path = vector_path
        # Converted the JavaScript this keyword to Python's self keyword, which refers to the current instance of the class.

    # }

    # // The main A* pathfinding algorithm
    # The main A* pathfinding algorithm
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

    # // References: https://en.wikipedia.org/wiki/A*_search_algorithm
    # References: https://en.wikipedia.org/wiki/A*_search_algorithm
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

    # // and https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
    # and https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

    # // g value is given by the distance from the start_node to current node
    # g value is given by the distance from the start_node to current node
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.

    # // h value is given by the Manhattan distance between current node and end_node
    # h value is given by the Manhattan distance between current node and end_node
    # Converted the comment from JavaScript to Python syntax by removing the double slashes (//) and using a single hash (#) for a Python comment.
    def astar(self, maze, start, end):
        # astar(maze, start, end) {
        # Converted the JavaScript method declaration to Python method declaration by adding the self parameter, which refers to the current instance of the class.

        # let start_node = new Node(start.x, start.y);
        start_node = Node(start["x"], start["y"])
        # Converted the JavaScript let keyword to Python's variable declaration without a keyword.
        # Converted the JavaScript new keyword to Python's class instantiation by directly calling the class name.
        # Converted the JavaScript dot notation (start.x) to Python's dictionary notation (start["x"]).

        # let end_node = new Node(end.x, end.y);
        end_node = Node(end["x"], end["y"])
        # Converted the JavaScript let keyword to Python's variable declaration without a keyword.
        # Converted the JavaScript new keyword to Python's class instantiation by directly calling the class name.
        # Converted the JavaScript dot notation (end.x) to Python's dictionary notation (end["x"]).

        # let open_list = [];
        open_list = []
        # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

        # let closed_list = [];
        closed_list = []
        # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

        # open_list.push(start_node);
        open_list.append(start_node)
        # Converted the JavaScript push() method to Python's append() method, which adds an element to the end of a list.

        # let possible_paths = [];
        possible_paths = []
        # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

        # const adjacent_squares = [
        adjacent_squares = [
            # Converted the JavaScript const keyword to Python's variable declaration without a keyword.
            [0, -1],
            [0, 1],
            [-1, 0],
            [1, 0],
        ]
        # No changes needed here, as the syntax for creating a list of lists in Python is the same as creating an array of arrays in JavaScript.

        # while (open_list.length > 0) {
        while len(open_list) > 0:
            # Converted the JavaScript while loop to Python's while loop.
            # Converted the JavaScript length property to Python's len() function, which returns the length of a list.

            # let current_node = open_list[0];
            current_node = open_list[0]
            # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

            # let current_index = 0;
            current_index = 0
            # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

            # let index = 0;
            index = 0
            # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

            # for (let i = 0; i < open_list.length; i++) {
            for i in range(len(open_list)):
                # Converted the JavaScript for loop to Python's for loop using the range() function and the len() function to get the length of the open_list.

                # if (open_list[i].f > current_node.f) {
                if open_list[i].f > current_node.f:
                    # No changes needed here, as the syntax is the same in both JavaScript and Python.

                    # current_node = open_list[i];
                    current_node = open_list[i]
                    # No changes needed here, as the syntax is the same in both JavaScript and Python.

                    # current_index = index;
                    current_index = index
                    # No changes needed here, as the syntax is the same in both JavaScript and Python.

                # }
                # index++;
                index += 1
                # Converted the JavaScript increment operator (++) to Python's increment operator (+=).

            # }

            # open_list.splice(current_index, 1);
            open_list.pop(current_index)
            # Converted the JavaScript splice() method to Python's pop() method, which removes an element at a specific index.

            # closed_list.push(current_node);
            closed_list.append(current_node)
            # Converted the JavaScript push() method to Python's append() method, which adds an element to the end of a list.

            # if (current_node.equals(end_node)) {
            if current_node.equals(end_node):
                # No changes needed here, as the syntax is the same in both JavaScript and Python.

                # let path = [];
                path = []
                # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

                # let current = current_node;
                current = current_node
                # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

                # while (current != null) {
                while current is not None:
                    # Converted the JavaScript null to Python's None keyword, which represents a null or undefined value.
                    # Converted the JavaScript inequality operator (!=) to Python's is not operator, which checks for identity rather than equality.

                    # path.push(current);
                    path.append(current)
                    # Converted the JavaScript push() method to Python's append() method, which adds an element to the end of a list.

                    # current = current.parent;
                    current = current.parent
                    # No changes needed here, as the syntax is the same in both JavaScript and Python.

                # }
                # possible_paths.push(path.reverse());
                possible_paths.append(list(reversed(path)))
                # Converted the JavaScript reverse() method to Python's reversed() function, which returns a reverse iterator.
                # Wrapped the reversed() function with list() to convert the iterator to a list.

            # }

            # let children = [];
            children = []
            # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

            # for (let i = 0; i < adjacent_squares.length; i++) {
            for i in range(len(adjacent_squares)):
                # Converted the JavaScript for loop to Python's for loop using the range() function and the len() function to get the length of the adjacent_squares.

                # let node_position = [current_node.x + adjacent_squares[i][0], current_node.y + adjacent_squares[i][1]];
                node_position = [
                    current_node.x + adjacent_squares[i][0],
                    current_node.y + adjacent_squares[i][1],
                ]
                # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

                # if (node_position[0] <= 39 && node_position[0] >= 0) {
                if 0 <= node_position[0] <= 39:
                    # Converted the JavaScript if statement to Python's if statement.
                    # Rearranged the condition to follow Python's comparison chaining, which allows multiple comparisons in a single expression.

                    # if (node_position[1] <= 19 && node_position[1] >= 0) {
                    if 0 <= node_position[1] <= 19:
                        # Converted the JavaScript if statement to Python's if statement.
                        # Rearranged the condition to follow Python's comparison chaining, which allows multiple comparisons in a single expression.

                        # if (maze[node_position[1]][node_position[0]] != -1) {
                        if maze[node_position[1]][node_position[0]] != -1:
                            # No changes needed here, as the syntax is the same in both JavaScript and Python.

                            # let new_node = new Node(node_position[0], node_position[1]);
                            new_node = Node(node_position[0], node_position[1])
                            # Converted the JavaScript let keyword to Python's variable declaration without a keyword.
                            # Converted the JavaScript new keyword to Python's class instantiation by directly calling the class name.

                            # children.push(new_node);
                            children.append(new_node)
                            # Converted the JavaScript push() method to Python's append() method, which adds an element to the end of a list.

                        # }
                    # }
                # }
            # }

            # for (let i = 0; i < children.length; i++) {
            for i in range(len(children)):
                # Converted the JavaScript for loop to Python's for loop using the range() function and the len() function to get the length of the children.

                # let if_in_closed_list = false;
                if_in_closed_list = False
                # Converted the JavaScript let keyword to Python's variable declaration without a keyword.
                # Converted the JavaScript false keyword to Python's False keyword, which represents a boolean false value.

                # for (let j = 0; j < closed_list.length; j++) {
                for j in range(len(closed_list)):
                    # Converted the JavaScript for loop to Python's for loop using the range() function and the len() function to get the length of the closed_list.

                    # if (children[i].equals(closed_list[j])) {
                    if children[i].equals(closed_list[j]):
                        # No changes needed here, as the syntax is the same in both JavaScript and Python.

                        # if_in_closed_list = true;
                        if_in_closed_list = True
                        # Converted the JavaScript true keyword to Python's True keyword, which represents a boolean true value.

                    # }
                # }
                # if (!if_in_closed_list) {
                if not if_in_closed_list:
                    # Converted the JavaScript logical NOT operator (!) to Python's logical NOT operator (not).

                    # children[i].g = current_node.g + 2;
                    children[i].g = current_node.g + 2
                    # No changes needed here, as the syntax is the same in both JavaScript and Python.

                    # children[i].h = abs(children[i].x - end_node.x) + abs(children[i].y - end_node.y);
                    children[i].h = abs(children[i].x - end_node.x) + abs(
                        children[i].y - end_node.y
                    )
                    # No changes needed here, as the syntax is the same in both JavaScript and Python.

                    # children[i].f = children[i].g + children[i].h;
                    children[i].f = children[i].g + children[i].h
                    # No changes needed here, as the syntax is the same in both JavaScript and Python.

                    # let present = false;
                    present = False
                    # Converted the JavaScript let keyword to Python's variable declaration without a keyword.
                    # Converted the JavaScript false keyword to Python's False keyword, which represents a boolean false value.

                    # for (let j = 0; j < open_list.length; j++) {
                    for j in range(len(open_list)):
                        # Converted the JavaScript for loop to Python's for loop using the range() function and the len() function to get the length of the open_list.

                        # if (children[i].equals(open_list[j]) && children[i].g < open_list[j].g) {
                        if (
                            children[i].equals(open_list[j])
                            and children[i].g < open_list[j].g
                        ):
                            # Converted the JavaScript logical AND operator (&&) to Python's logical AND operator (and).

                            # present = true;
                            present = True
                            # Converted the JavaScript true keyword to Python's True keyword, which represents a boolean true value.

                        # } else if (children[i].equals(open_list[j]) && children[i].g >= open_list[j].g) {
                        elif (
                            children[i].equals(open_list[j])
                            and children[i].g >= open_list[j].g
                        ):
                            # Converted the JavaScript else if statement to Python's elif statement.
                            # Converted the JavaScript logical AND operator (&&) to Python's logical AND operator (and).

                            # open_list[j] = children[i];
                            open_list[j] = children[i]
                            # No changes needed here, as the syntax is the same in both JavaScript and Python.

                            # open_list[j].parent = current_node;
                            open_list[j].parent = current_node
                            # No changes needed here, as the syntax is the same in both JavaScript and Python.

                        # }
                    # }
                    # if (!present) {
                    if not present:
                        # Converted the JavaScript logical NOT operator (!) to Python's logical NOT operator (not).

                        # children[i].parent = current_node;
                        children[i].parent = current_node
                        # No changes needed here, as the syntax is the same in both JavaScript and Python.

                        # open_list.push(children[i]);
                        open_list.append(children[i])
                        # Converted the JavaScript push() method to Python's append() method, which adds an element to the end of a list.

                    # }
                # }
            # }
        # }
        # let path = [];
        path = []
        # Converted the JavaScript let keyword to Python's variable declaration without a keyword.

        # for (let i = 0; i < possible_paths.length; i++) {
        for i in range(len(possible_paths)):
            # Converted the JavaScript for loop to Python's for loop using the range() function and the len() function to get the length of the possible_paths.

            # if (possible_paths[i].length > path.length) {
            if len(possible_paths[i]) > len(path):
                # Converted the JavaScript length property to Python's len() function, which returns the length of a list.

                # path = possible_paths[i];
                path = possible_paths[i]
                # No changes needed here, as the syntax is the same in both JavaScript and Python.

            # }
        # }
        # return path;
        return path
        # No changes needed here, as the syntax is the same in both JavaScript and Python.

    # }


# }
