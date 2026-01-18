// Main package for the Go project
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
)

// Result represents the output of the run function
type Result struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

// Run executes the main functionality of the project
func Run() Result {
	return Result{
		Status:  "success",
		Message: "Hello from Go project!",
	}
}

func main() {
	result := Run()
	
	// Convert result to JSON
	jsonData, err := json.Marshal(result)
	if err != nil {
		log.Fatalf("Error marshalling result: %v", err)
	}
	
	fmt.Printf("Result: %s\n", string(jsonData))
	os.Exit(0)
}
