#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

class Tokenizer {
private:
    std::unordered_map<std::string, int> stringToId;
    std::unordered_map<int, std::string> idToString;
    int nextId = 0;

public:
    int getOrAdd(const std::string& text) {
        auto it = stringToId.find(text);
        if (it != stringToId.end()) {
            return it->second;
        }

        int newId = nextId++;
        stringToId[text] = newId;
        idToString[newId] = text;
        return newId;
    }

    std::string getString(int id) const {
        auto it = idToString.find(id);
        if (it != idToString.end()) {
            return it->second;
        }
        throw std::out_of_range("ID inconnu dans le tokenizer.");
    }

    int size() const {
        return nextId;
    }
};