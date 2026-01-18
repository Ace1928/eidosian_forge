//! Main entry point for the Rust project.

use serde::{Deserialize, Serialize};

/// Result of the run function
#[derive(Debug, Serialize, Deserialize)]
pub struct RunResult {
    status: String,
    message: String,
}

/// Run the main functionality of the project
pub fn run() -> RunResult {
    RunResult {
        status: String::from("success"),
        message: String::from("Hello from Rust project!"),
    }
}

fn main() {
    let result = run();
    println!("Result: {:?}", result);
}
