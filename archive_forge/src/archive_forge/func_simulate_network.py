def simulate_network(input_signals, time_steps):
    neuron_outputs = {node_id: np.zeros(time_steps) for node_id in G.nodes}
    feedback_signals = {node_id: np.zeros(time_steps) for node_id in G.nodes}
    for t in range(time_steps):
        for node_id in range(1, 8):
            neuron = neurons[node_id]
            input_signal = input_signals.get(node_id, [0] * time_steps)[t]
            neuron.process_input_with_subthreshold_oscillation(input_signal, t)
            neuron_outputs[node_id][t] = neuron.output
            feedback_signals[node_id][t] = neuron.feedback_signal
        for (source, target), connection in connections.items():
            delay = connection.delay
            propagated_time = t - delay
            if propagated_time >= 0:
                propagated_signal = neuron_outputs[source][propagated_time] * connection.strength
                neurons[target].process_input_with_subthreshold_oscillation(propagated_signal, t)
    return (neuron_outputs, feedback_signals)