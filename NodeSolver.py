import socket
import threading
import logging
import pickle
from collections import deque
import argparse
import json
import random
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from itertools import permutations
import math
import uuid

from sudoku import Sudoku

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SudokuSolver:
    def __init__(self, base_delay=0.01):
        logger.info("Sudoku Solver initialized")
        self.validations = 0
        self.base_delay = base_delay
        self.all_range_index = []

    def permut(self, index, values, save_numbers, not_values_line):
        result = []
        for first, rest in [(values[i], values[:i] + values[i+1:]) for i in range(len(values))]:
            for p in permutations(rest):
                aux = [first] + list(p)
                for key, value in save_numbers.items():
                    aux.insert(key, value)
                if self.valid_line_values_coluns(index, aux, not_values_line):
                    result.append(aux)
        return result

    def valid_line_values_coluns(self, index, line, vals):
        for c in range(len(line)):
            value = line[c]
            if index in vals and c in vals[index] and value == vals[index][c]:
                continue
            else:
                for aux in vals:
                    if c in vals[aux] and value == vals[aux][c]:
                        return False
        return True  

    def line_combinations(self, board):
        
        not_values_line = self.used_values_col(board)
        all_combinations = []

        for index in range(len(board)):
            line = board[index]
            values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            save_numbers = {}
            for i, c in enumerate(line):
                if c != 0:
                    save_numbers[i] = c
                    values.pop(values.index(c))

            aux = list(self.permut(index, values, save_numbers, not_values_line))
            if len(aux) == 0:
                aux = [line]
            all_combinations.append(aux)

        return all_combinations

    def validate(self, sdk):
        if len(sdk) < 2:
            return True
        
        for c in range(9):
            col = []
            for line in sdk:
                if line[c] in col:
                    return False
                col.append(line[c])

        if len(sdk) > 2:
            for i in range(len(sdk) // 3):
                for j in range(3):
                    if sum([sdk[i*3+k][j*3+l] for k in range(3) for l in range(3)]) != 45:
                        return False

        return True

    def combin_lines(self, all_combinations, range_number, counter, next_line_index=None):
        
        print("COMBIN LINES: ", range_number)
        
        size = 5
        for c in all_combinations:
            print("COMBIN LINES: ", len(c))
        all_comb2 = [list(comb) for comb in all_combinations]
        index_filter = 0
        size_line_cominations = 0

        print("all_comb2: ", len(all_comb2))
        if len(all_comb2) < 120:
            for c in range(len(all_comb2)):
                if size_line_cominations < len(all_comb2[c]):
                    size_line_cominations = len(all_comb2[c])
                    index_filter = c


        if counter == 0:    
            combin_line1 = len(all_comb2[index_filter])
            print(combin_line1)
            num_of_ranges = combin_line1 // size
            print("NUM OF RANGES: ", num_of_ranges)
            if combin_line1 % size != 0:
                num_of_ranges += 1
            self.all_range_index = list(range(next_line_index, num_of_ranges))
        else:
            self.all_range_index = next_line_index
            
        print("WTF: ",self.all_range_index)
        sdk = []

        start_index = size * range_number
        end_index = start_index + size  
        # print("all_comb2[index_filter]: ", len(all_comb2[index_filter]))
        print("start_index: ", start_index)
        print("end_index: ", end_index)
        if start_index < len(all_comb2[index_filter]):
            if end_index > len(all_comb2[index_filter]):
                end_index = len(all_comb2[index_filter])
            selected_combinations = all_comb2[index_filter][start_index:end_index]
            
            all_comb2[index_filter] = selected_combinations
            # print("JESUSKID: ", all_comb2[index_filter])
            # print("all_comb2[index_filter]: ", len(all_comb2[index_filter]))

        if len(all_comb2[index_filter]) == 0:
            print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
            print("VALIDATIONS: ", self.validations)
            return []



        def backtrack(index):
            if index == len(all_comb2):
                return True 
            
            for line in all_comb2[index]:

                sdk.append(line)
                sdk_aux = Sudoku(sdk)
                
                print(sdk)
                if len(sdk) != 9: 
                    self.validations += 1

                    if self.validate(sdk):
                        if backtrack(index + 1):
                            return True
                else:
                    self.validations += 1
                    if sdk_aux.check():
                        if backtrack(index + 1):
                            return True

                
                sdk.pop() 
            
            return False

        if backtrack(0):
            print("VALIDATIONS: ", self.validations)
            return sdk, self.all_range_index
        else:
            
            print("NÃO DEU CERTO")
            print("VALIDATIONS: ", self.validations)
            
            return None, self.all_range_index

    def used_values_col(self, sdk):
        values_in_use_col = {i: {} for i in range(9)}

        for row_idx, line in enumerate(sdk):
            for col_idx, val in enumerate(line):
                if val != 0:
                    values_in_use_col[row_idx][col_idx] = val
        
        a = {}
        for k in list(values_in_use_col.keys()):
            if values_in_use_col[k] == a:
                values_in_use_col.pop(k)
        
        return values_in_use_col  


class P2PNode(threading.Thread):
    def __init__(self, host, port, peer_address=None, handicap=0.001):
        super().__init__()
        self.solver = SudokuSolver()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.id = f"{host}:{port}"
        self.host = host
        self.port = port

        self.all_combinations = None
        
        self.peer_address = peer_address
        self.shutdown_flag = False
        self.solved = True

        self.all_range_index = []

        self.current_range_number = None

        self.counter = 0
        self.solved_puzzles = 0

        self.sudoku = None
        self.initial_sudoku = None
        self.partial_solution = None

        self.total_peers = set()
        self.all_peers = {}
        self.peers_to_reconnect = {}

        self.handicap = handicap
        self.range_number = 0
        self.resolving_peer = None
        self.resolving_sudoku = None
        self.resolving_addr = None
        self.active_tasks = {}
        self.solution_queue = deque()
        self.task_queue = deque()
        self.tried_numbers_by_position = {}
        self.stats_solved = {}
        self.all_stats = {
            "all": {
                "solved": 0,
                "validations": 0
            },
            "nodes": [
                {
                    "address": self.id,
                    "validations": 0
                }
            ]
        }

        self.validations = 0

        self.high_priority_queue = deque()

    

    def update_validations_solution(self, adress, validations):
        all_validations = 0
        

        for node in self.all_stats["nodes"]:
            if node["address"] == adress:
                
                node["validations"] = validations
                print("NODE_VAL: ", validations)
                print("FOR: ", self.id)
                print("NODE: ", node)
 
            all_validations += node["validations"]
                
        self.all_stats["all"]["validations"] = all_validations

        print("UPDATED STATS: ", self.all_stats)
        


    def send(self, msg, address=None):
        """Send msg to address, or broadcast if address is None."""
        payload = json.dumps(msg).encode()
        if address:
            self.sock.sendto(payload, address)
        else:
            self.broadcast(msg)
        

    def broadcast(self, msg):
        """Broadcast msg to all peers."""
        for peer in self.all_peers:
            host, port = peer.split(':')
            self.send(msg, (host, int(port)))

    def recv(self):
        """Retrieve msg payload and from address."""
        try:
            payload, addr = self.sock.recvfrom(1024)
            if len(payload) == 0:
                return None, addr
            return payload, addr
        except socket.timeout:
            return None, None
        except Exception as e:
            logging.error(f"Error receiving data: {e}")
            return None, None

    def handle_message(self, msg):
        logging.info(f"Node {self.id} received message: {msg}")
        

        if msg['type'] == 'join':
            new_peer = msg['address']
            print("DAMN: ", new_peer)
            self.update_peer_list(new_peer)
            self.announce_all_peers(new_peer)
            print("JOAO: ",self.solved)
            # print("JOAO: ",self.id)
            # print(self.solved)
            logging.info(f"Processed join message from {new_peer}")
            


        elif msg['type'] == 'new_peer':
            self.all_stats = msg['all_stats']
            print("ALL STATS: ", self.all_stats)
            self.solved = msg['solved']
            self.all_peers = msg['all_pears']
            if self.solved == False:
                self.all_range_index = msg['all_range_index']
                new_peer = msg['new_peer']
                self.sudoku = msg['sudoku']
                if self.id == new_peer:
                    self.solved = msg['solved']
                    msg = {"type": "update_range_number", "from": self.id, "is_new_peer": True}
                    self.send(msg)

            logging.info(f"Node {self.id} received new peer list: {self.all_peers}")
            
            

        elif msg['type'] == 'solution':
            self.counter == 0
            self.all_stats["all"]["solved"] += 1
           
            self.solved = True
            print("RECEIVED SOLUTION FROM: ", msg['from'])
            
            self.sudoku = msg['sudoku']

            for peer in self.all_peers:
                host, port = peer.split(':')
                msg = {'type': 'stats', 'validations': self.solver.validations, 'address': self.id}
                self.send(msg, (host, int(port)))
                time.sleep(0.5)
           
            


        elif msg['type'] == 'update_range_number':
            is_new_peer = msg['is_new_peer']
            if len(self.all_range_index) > 0 and self.solved == False:
                print("ALL RANGE INDEX: ", self.all_range_index)
                self.range_number = self.all_range_index.pop(0)
                
                if self.id == msg['from']:
                    if is_new_peer:
                        next_line_index = self.range_number + 1
                        msg = {"sudoku": self.sudoku, 'range_number': self.range_number, 'next_line_index': next_line_index, 'is_new': True}
                        self.process_solve_message(msg)

                    else:
                        print("DEIIIIIII UPDAATEEEEE FROM: ", msg['from'])
                        self.process_solve_message(self.range_number)
            else:
                self.solved = False
                

        elif msg['type'] == 'disconnect':
            logging.info(f"Received disconnect message from {msg['address']}")
            print(f"Message contents: {msg}")
            if 'all_range_index' in msg:
                print("TEM ALL RANGE INDEX")
                self.all_range_index = msg['all_range_index']

            self.all_peers.pop(msg['address'], None)
            for peer in self.all_peers.values():
                if msg['address'] in peer:
                    peer.remove(msg['address'])

            for peer in self.all_peers:
                host, port = peer.split(':')
                msg = {'type': 'stats', 'validations': msg['validations'], 'address': msg['address']}
                self.send(msg, (host, int(port)))
                time.sleep(0.5)

           
            

        elif msg['type'] == 'stats':
            logging.info("Received stats message")
            self.update_validations_solution(msg['address'], msg['validations'])
            print("UPDATED_STATS: ", self.all_stats)
            
            



        elif msg['type'] == 'solve':
            self.counter = 0
            self.process_solve_message(msg)
            print("OLA: ", self.counter)

        
            
            
        
        





    def process_solve_message(self, msg):
        logging.info("Processing solve")
        print(self.solved)

        logging.info("Processing solve message, attempt %d", self.counter)

        
        if self.counter == 0:
            self.sudoku = msg['sudoku']
            self.range_number = msg['range_number']
            next_line_index = msg['next_line_index']
            self.current_range_number = self.range_number
            print("SUDOKU: ", self.sudoku)

            self.all_combinations = self.solver.line_combinations(self.sudoku)  
            if 'is_new' in msg:
                self.resolving_peer = True
                sdk, all_range_indexes = self.solver.combin_lines(self.all_combinations, self.range_number, self.counter, next_line_index)
                
            else:
                self.resolving_peer = True
                sdk, all_range_indexes = self.solver.combin_lines(self.all_combinations, self.range_number, self.counter, next_line_index)
                
                self.all_range_index = all_range_indexes
        
        else:
            self.resolving_peer = True
            self.range_number = msg
            self.current_range_number = self.range_number
            print("NEXT RANGE NUMBER: ", self.range_number)
            print("ESTOU AQUI: ", len(self.all_combinations[0]))
            sdk, all_range_indexes = self.solver.combin_lines(self.all_combinations, self.range_number, self.counter, self.all_range_index)
            
            

        
        
        

        
        if sdk == []:
            self.resolving_peer = False
            logging.info("No more intervals to search")
            self.current_range_number = None
            self.solved = False
            
        elif sdk is not None:
            self.resolving_peer = False
            sdk = Sudoku(sdk)
            if sdk.check(base_delay=self.handicap):
                logging.info("Sudoku is correct!")
                
                msg = {"type": "solution", "sudoku": sdk.grid, "from": self.id}
                self.send(msg)


                    
                
                print("FINAL STATS: ", self.all_stats)
                self.current_range_number = None
            else:
                self.solved = False
                logging.info("Sudoku is incorrect, please check your solution.")
        else:
            self.resolving_peer = False
            logging.info("No more combinations to search for this interval")
            self.solved = False
            self.counter += 1
            if self.solved == False:
                self.current_range_number = None
                msg = {"type": "update_range_number", "from": self.id, "is_new_peer": False}
                self.send(msg)
                



    def update_peer_list(self, new_peer):
        new_node = {
        "address": new_peer,
        "validations": 0
        }
        self.all_stats["nodes"].append(new_node)
        
        for peer in self.all_peers:
            self.all_peers[peer].append(new_peer)
        self.all_peers[new_peer] = [p for p in self.all_peers if p != new_peer]

    def announce_all_peers(self, new_peer=None):
        if self.solved == False:
            msg = {"type": "new_peer", "all_pears": self.all_peers, "all_range_index": self.all_range_index, "solved": self.solved, "new_peer": new_peer, "sudoku": self.sudoku, 'all_stats': self.all_stats}
        else:
            msg = {"type": "new_peer", "all_pears": self.all_peers, "solved": self.solved, 'all_stats': self.all_stats}
        self.send(msg)

    

    
                                                                                               
    def find_solution(self, sudoku):
        self.solved = False
        self.sudoku = sudoku
        self.range_number = 0
        

        for peer in self.all_peers:
            
            print("SENDING TO: ", peer)
            next_line_index = len(self.all_peers)
            msg = {"type": "solve", "sudoku": sudoku, 'range_number': self.range_number, 'next_line_index': next_line_index}
            host, port = peer.split(':')
            self.send(msg, (host, int(port)))

            self.range_number += 1
            

        while self.solved == False:
            pass        
        return self.sudoku
    
    def get_stats(self):
        msg = {"type": "stats"}
        
        return self.all_stats




    def run(self):
        """Run the P2P server."""
        self.sock.bind((self.host, self.port))
        logging.info(f"P2P Server with id {self.id} listening on {self.host}:{self.port}")
        if self.peer_address:
            if self.peer_address not in self.all_peers:
                self.all_peers[self.peer_address] = []  # Initialize list if not exists
            join_msg = {"type": "join", "address": self.id}
            host, port = self.peer_address.split(':')
            self.send(join_msg, (host, int(port)))
        else:
            self.all_peers[self.id] = []
        
        while not self.shutdown_flag:
            try:
                payload, addr = self.recv()


                if payload is not None:
                    msg = json.loads(payload.decode())
                    if msg['type'] == 'join':
                        self.high_priority_queue.append(msg)
                    else:
                        self.task_queue.append(msg)

                if self.high_priority_queue:
                    high_priority_msg = self.high_priority_queue.popleft()
                    self.handle_message(high_priority_msg)
                elif self.task_queue:
                    msg = self.task_queue.popleft()
                    self.handle_message(msg)
                    
                time.sleep(self.handicap)



            except KeyboardInterrupt:
                self.shutdown()
            except Exception as e:
                logging.error(f"Error receiving data: {e}")
                continue

    def shutdown(self):
        
        self.shutdown_flag = True

        if self.current_range_number is not None:  # Check if there's an active range being processed
            self.all_range_index.append(self.current_range_number)
            self.current_range_number = None

        validations = self.solver.validations
        print("LOL: ", validations)

        msg = {"type": "disconnect", "address": self.id, "all_range_index": self.all_range_index, 'validations': validations}
        if self.resolving_peer == True:
            
            msg["all_range_index"] = self.all_range_index
        for peer in list(self.all_peers.keys()):
            if peer != self.id:
                
                
                self.send(msg)
                logging.info(f"Sent disconnect message to {peer}")

        logging.info(f"Shutting down P2P node id {self.id}")


class SudokuRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, p2p_node, *args, **kwargs):
        self.p2p_node = p2p_node
        super().__init__(*args, **kwargs)

    def _send_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_POST(self):
        if self.path == '/solve':
            initial_time = time.time()
            logging.info("Received POST request to solve Sudoku")
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            sudoku = json.loads(post_data.decode('utf-8'))['sudoku']
            solution = self.p2p_node.find_solution(sudoku)

            final_time = time.time()
            elapsed_time = final_time - initial_time
            logging.info(f"Execution time {elapsed_time}")

            
            if solution is not None:
                self._send_response(solution)
            else:
                self._send_response({"error": "Failed to solve Sudoku puzzle"}, 500)
        else:
            self._send_response({"error": "Invalid path"}, 404)

    def do_GET(self):
        if self.path == '/stats':
            logging.info("Received GET request for stats")
            self._send_response(self.p2p_node.get_stats())

        elif self.path == '/network':
            logging.info("Received GET request for network")
            if self.p2p_node.all_peers:
                self._send_response(self.p2p_node.all_peers)
            else:
                dic = {self.p2p_node.id: []}
                self._send_response(dic)
        else:
            self._send_response({"error": "Invalid path"}, 404)

def run_http_server(p2p_node, port):
    host = 'localhost'
    server_address = (host, port)
    httpd = HTTPServer(server_address, lambda *args, **kwargs: SudokuRequestHandler(p2p_node, *args, **kwargs))
    logging.info(f'Starting HTTP server on {host} port {port}')
    httpd.serve_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed Sudoku Solver')
    parser.add_argument('-p', type=int, required=True, help='HTTP server port')
    parser.add_argument('-s', type=int, required=True, help='P2P server port')
    parser.add_argument('-a', type=str, help='Address of a peer to join')
    parser.add_argument('-d', type=int, default=0, help='Handicap in milliseconds')

    args = parser.parse_args()

    http_port = args.p
    p2p_port = args.s
    peer_address = args.a
    handicap = args.d

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    p2p_node = P2PNode("localhost", p2p_port, peer_address, handicap/100)

    http_thread = threading.Thread(target=run_http_server, args=(p2p_node, http_port))
    http_thread.daemon = True
    http_thread.start()
    p2p_node.run()


