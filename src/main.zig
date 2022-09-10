const std = @import("std");
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;
const GeneralPurposeAllocator = std.heap.GeneralPurposeAllocator;

var gpa = GeneralPurposeAllocator(.{}){};
var arena: ArenaAllocator = undefined;

pub fn main() anyerror!void {
    arena = ArenaAllocator.init(gpa.allocator());
}

pub fn log(
    comptime _: std.log.Level,
    comptime _: @TypeOf(.EnumLiteral),
    comptime _: []const u8,
    _: anytype,
) void {}

pub const os = struct {
    pub const system = struct {
        pub extern fn exit(status: u8) noreturn;
    };
};

fn throwError() void {
    std.process.exit(1);
}

fn signalError(message: []const u8, what_: ?[]const u8) void {
    if (what_) |what| {
        std.log.err("error: {s}: {s}\n", .{ message, what });
    } else {
        std.log.err("error: {s}\n", .{message});
    }
    throwError();
}

const Expr = struct {
    const Kind = enum {
        Func,
        LetRec,
        Var,
        Prim,
        Literal,
        Call,
        If,
    };

    kind: Kind,

    fn init(kind: Kind) Expr {
        return .{ .kind = kind };
    }
};

const Func = struct {
    const arg_count: u32 = 1;

    expr: Expr, // base class
    body: *const Expr, // unique_ptr
    jit_code: ?*anyopaque, // void *

    fn offsetOfBody() usize {
        return @offsetOf(Func, "body");
    }

    fn offsetOfJitCode() usize {
        return @offsetOf(Func, "jit_code");
    }

    fn init(body: *const Expr) Func {
        return .{
            .expr = Expr.init(Expr.Kind.Func),
            .body = body,
            .jit_code = null,
        };
    }
};

const LetRec = struct {
    const arg_count: u32 = 1;

    expr: Expr, // base class
    arg: *const Expr, // unique_ptr
    body: *const Expr, // unique_ptr

    fn init(arg: *const Expr, body: *const Expr) LetRec {
        return .{
            .expr = Expr.init(Expr.Kind.LetRec),
            .arg = arg,
            .body = body,
        };
    }
};

const Var = struct {
    expr: Expr, // base class
    depth: u32,

    fn init(depth: u32) Var {
        return .{
            .expr = Expr.init(Expr.Kind.Var),
            .depth = depth,
        };
    }
};

const Prim = struct {
    const Op = enum {
        Eq,
        LessThan,
        Sub,
        Add,
        Mul,
    };

    expr: Expr, // base class
    op: Op,
    lhs: *const Expr, // unique_ptr
    rhs: *const Expr, // unique_ptr

    fn init(op: Op, lhs: *const Expr, rhs: *const Expr) Prim {
        return .{
            .expr = Expr.init(Expr.Kind.Prim),
            .op = op,
            .lhs = lhs,
            .rhs = rhs,
        };
    }
};

const Literal = struct {
    expr: Expr, // base class
    val: i32,

    fn init(val: i32) Literal {
        return .{
            .expr = Expr.init(Expr.Kind.Literal),
            .val = val,
        };
    }
};

const Call = struct {
    expr: Expr, // base class
    func: *const Expr, // unique_ptr
    arg: *const Expr, // unique_ptr

    fn init(func: *const Expr, arg: *const Expr) Call {
        return .{
            .expr = Expr.init(Expr.Kind.Call),
            .func = func,
            .arg = arg,
        };
    }
};

const If = struct {
    expr: Expr, // base class
    test_: *const Expr, // unique_ptr
    consequent: *const Expr, // unique_ptr
    alternate: *const Expr, // unique_ptr

    fn init(test_: *const Expr, consequent: *const Expr, alternate: *const Expr) If {
        return .{
            .expr = Expr.init(Expr.Kind.If),
            .test_ = test_,
            .consequent = consequent,
            .alternate = alternate,
        };
    }
};

const Parser = struct {
    bound_vars: std.ArrayList([]const u8),
    buf: [*:0]const u8,
    pos: usize,
    len: usize,

    fn isAlphabetic(c: u8) bool {
        return ('a' <= c and c <= 'z') or ('A' <= c and c <= 'Z');
    }

    fn isNumeric(c: u8) bool {
        return '0' <= c and c <= '9';
    }

    fn isAlphaNumeric(c: u8) bool {
        return isAlphabetic(c) or isNumeric(c);
    }

    fn isWhitespace(c: u8) bool {
        return c == ' ' or c == '\t' or c == '\n' or c == '\r';
    }

    fn init(allocator: Allocator, buf: [*:0]const u8) Parser {
        return .{
            .bound_vars = std.ArrayList([]const u8).init(allocator),
            .buf = buf,
            .pos = 0,
            .len = std.mem.indexOfSentinel(u8, 0, buf),
        };
    }

    fn parse(self: *Parser) ?*const Expr {
        const e = self.parseOne();
        self.skipWhitespace();
        if (!self.eof())
            self.err("expected end of input after expression");
        return e;
    }

    fn pushBound(self: *Parser, id: []const u8) void {
        self.bound_vars.append(id) catch std.process.exit(1);
    }

    fn popBound(self: *Parser) void {
        _ = self.bound_vars.pop();
    }

    fn lookupBound(self: Parser, id: []const u8) u32 {
        var i: usize = 0;
        while (i < self.bound_vars.items.len) : (i += 1) {
            if (self.bound_vars.items[self.bound_vars.items.len - i - 1].ptr == id.ptr)
                return i;
        }
        signalError("unbound identifier", id);
        return @bitCast(u32, @as(i32, -1));
    }

    fn err(self: Parser, message: []const u8) void {
        signalError(message, if (self.eof()) self.buf[0..self.len] else self.buf[self.pos..self.len]);
    }

    fn eof(self: Parser) bool {
        return self.pos == self.len;
    }

    fn peek(self: Parser) u8 {
        if (self.eof()) return 0;
        return self.buf[self.pos];
    }

    fn advance(self: *Parser) void {
        if (!self.eof()) self.pos += 1;
    }

    fn next(self: *Parser) u8 {
        const ret = self.peek();
        self.advance();
        return ret;
    }

    fn matchChar(self: *Parser, c: u8) bool {
        if (self.eof() or self.peek() != c)
            return false;
        self.advance();
        return true;
    }

    fn skipWhitespace(self: *Parser) void {
        while (!self.eof() and isWhitespace(self.peek()))
            self.advance();
    }

    fn startsIdentifier(self: Parser) bool {
        return !self.eof() and isAlphabetic(self.peek());
    }

    fn continuesIdentifier(self: Parser) bool {
        return !self.eof() and isAlphaNumeric(self.peek());
    }

    fn matchIdentifier(self: *Parser, literal: []const u8) bool {
        const match_len = literal.len;
        if (match_len < (self.len - self.pos))
            return false;
        // if (strncmp(self.buf + self.pos, literal, match_len) != 0)
        if (!std.mem.eql(u8, self.buf[self.pos .. self.pos + self.len], literal))
            return false;
        if ((self.len - self.pos) < match_len and isAlphaNumeric(self.buf[self.pos + match_len]))
            return false;
        self.pos += match_len;
        return true;
    }

    fn takeIdentifier(self: *Parser) []const u8 {
        const start = self.pos;
        while (self.continuesIdentifier())
            self.advance();
        const end = self.pos;
        return self.buf[start..end];
    }

    fn matchKeyword(self: *Parser, kw: []const u8) bool {
        const kwlen = kw.len;
        var remaining = self.len - self.pos;
        if (remaining < kwlen)
            return false;
        if (!std.mem.eql(u8, self.buf[self.pos .. self.pos + kwlen], kw))
            return false;
        self.pos += kwlen;
        if (!self.continuesIdentifier())
            return true;
        self.pos -= kwlen;
        return false;
        // if ((self.len - self.pos) < kwlen and isalnum(self.buf[self.pos + kwlen]))
        //   return 0;
        // self.pos += kwlen;
        // return 1;
    }

    fn parsePrim(self: *Parser, op: Prim.Op) *const Expr {
        const lhs = self.parseOne();
        const rhs = self.parseOne();
        return &Prim.init(op, lhs.?, rhs.?).expr;
    }

    fn parseInt32(self: *Parser) i32 {
        var ret: u64 = 0;
        while (!self.eof() and isNumeric(self.peek())) {
            ret *= 10;
            ret += self.next() - '0';
            if (ret > 0x7fffffff)
                self.err("integer too long");
        }
        if (!self.eof() and !isWhitespace(self.peek()) and self.peek() != ')')
            self.err("unexpected integer suffix");
        return @intCast(i32, ret);
    }

    fn parseOne(self: *Parser) ?*const Expr {
        self.skipWhitespace();
        if (self.eof())
            self.err("unexpected end of input");
        if (self.matchChar('(')) {
            self.skipWhitespace();

            var ret: *const Expr = undefined;
            if (self.matchKeyword("lambda")) {
                self.skipWhitespace();
                if (!self.matchChar('('))
                    self.err("expected open paren after lambda");
                self.skipWhitespace();
                if (!self.startsIdentifier())
                    self.err("expected an argument for lambda");
                self.pushBound(self.takeIdentifier());
                self.skipWhitespace();
                if (!self.matchChar(')'))
                    self.err("expected just one argument for lambda");
                const body = self.parseOne();
                self.popBound();
                ret = &Func.init(body.?).expr;
            } else if (self.matchKeyword("letrec")) {
                self.skipWhitespace();
                if (!self.matchChar('('))
                    self.err("expected open paren after letrec");
                if (!self.matchChar('('))
                    self.err("expected two open parens after letrec");
                self.skipWhitespace();
                if (!self.startsIdentifier())
                    self.err("expected an identifier for letrec");
                self.pushBound(self.takeIdentifier());
                self.skipWhitespace();
                const arg = self.parseOne();
                if (!self.matchChar(')'))
                    self.err("expected close paren after letrec binding");
                self.skipWhitespace();
                if (!self.matchChar(')'))
                    self.err("expected just one binding for letrec");
                const body = self.parseOne();
                self.popBound();
                ret = &LetRec.init(arg.?, body.?).expr;
            } else if (self.matchKeyword("+")) {
                ret = self.parsePrim(Prim.Op.Add);
            } else if (self.matchKeyword("-")) {
                ret = self.parsePrim(Prim.Op.Sub);
            } else if (self.matchKeyword("<")) {
                ret = self.parsePrim(Prim.Op.LessThan);
            } else if (self.matchKeyword("eq?")) {
                ret = self.parsePrim(Prim.Op.Eq);
            } else if (self.matchKeyword("*")) {
                ret = self.parsePrim(Prim.Op.Mul);
            } else if (self.matchKeyword("if")) {
                const test_ = self.parseOne();
                const consequent = self.parseOne();
                const alternate = self.parseOne();
                ret = &If.init(test_.?, consequent.?, alternate.?).expr;
            } else {
                // Otherwise it's a call.
                const func = self.parseOne();
                const arg = self.parseOne();
                ret = &Call.init(func.?, arg.?).expr;
            }
            self.skipWhitespace();
            if (!self.matchChar(')'))
                self.err("expected close parenthesis");
            return ret;
        } else if (self.startsIdentifier()) {
            return &Var.init(self.lookupBound(self.takeIdentifier())).expr;
        } else if (isNumeric(self.peek())) {
            return &Literal.init(self.parseInt32()).expr;
        }
        self.err("unexpected input");
        return null;
    }
};

export fn parse(str: [*:0]const u8) ?*const Expr {
    var parser = Parser.init(arena.allocator(), str);
    return parser.parse();
}

export fn allocateBytes(len: usize) *anyopaque {
    return (gpa.allocator().alloc(u8, len) catch std.process.exit(1)).ptr;
}
