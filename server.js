import "dotenv/config";
import express from "express";
import cors from "cors";
import OpenAI from "openai";
import multer from "multer";
import XLSX from "xlsx";
import { HfInference } from "@huggingface/inference";
import admin from "firebase-admin";

const type= process.env.FIREBASE_TYPE;
const project_id= process.env.FIREBASE_PROJECT_ID;
const private_key_id= process.env.FIREBASE_PRIVATE_KEY_ID;
const private_key= process.env.FIREBASE_PRIVATE_KEY;
const client_email= process.env.FIREBASE_CLIENT_EMAIL;
const client_id= process.env.FIREBASE_CLIENT_ID;
const auth_uri= process.env.FIREBASE_AUTH_URI;
const token_uri= process.env.FIREBASE_TOKEN_URI;
const auth_provider_x509_cert_url= process.env.FIREBASE_AUTH_PROVIDER_X509_CERT_URL;
const client_x509_cert_url= process.env.FIREBASE_CLIENT_X509_CERT_URL;
const universe_domain= process.env.FIREBASE_UNIVERSE_DOMAIN;
// Initialize Firebase Admin SDK
admin.initializeApp({
  credential: admin.credential.cert({
    "type": type,
    "project_id": project_id,
    "private_key_id": private_key_id,
    "private_key": private_key,
    "client_email": client_email,
    "client_id": client_id,
    "auth_uri": auth_uri,
    "token_uri": token_uri,
    "auth_provider_x509_cert_url": auth_provider_x509_cert_url,
    "client_x509_cert_url": client_x509_cert_url,
    "universe_domain": universe_domain
  }
  ),
});

const db = admin.firestore();

const app = express();
const PORT = process.env.PORT || 5000;

const apiToken = process.env.HUGGINGFACE_API_TOKEN;
const client = new HfInference(apiToken);

const deepseek = new OpenAI({
  baseURL: "https://api.deepseek.com",
  apiKey: process.env.DEEPSEEK_API_KEY,
});

app.use(express.json());
app.use(cors( ));

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, __dirname); // Save the file in the server's root directory
  },
  filename: function (req, file, cb) {
    cb(null, "users.xlsx"); // Save the file with a fixed name
  },
});

const upload = multer({ storage: storage });

// Helper function to handle Firestore operations
const handleFirestoreError = (res, error) => {
  console.error("Firestore Error:", error);
  res.status(500).json({ error: "Firestore operation failed" });
};

// 2. Chat related functions
// Store chat messages
app.post("/api/storeChat", async (req, res) => {
  const { chatId, messages, chatName, userId } = req.body;
  console.log("Received storeChat request:", req.body); // Log the request body

  try {
    const chatRef = db.collection("chats").doc(chatId);
    const chatDoc = await chatRef.get();

    if (!chatDoc.exists) {
      return res.status(404).json({ error: "Chat not found" });
    }

    // Only update the messages field, and keep the existing chatName
    await chatRef.update({
      messages,
    });

    res.status(200).json({ message: "Chat stored successfully" });
  } catch (error) {
    handleFirestoreError(res, error);
  }
});

// Get all chats
app.get("/api/getChats", async (req, res) => {
  try {
    const chatsSnapshot = await db.collection("chats").get();
    const chats = chatsSnapshot.docs.map((doc) => doc.data());
    res.status(200).json(chats);
  } catch (error) {
    handleFirestoreError(res, error);
  }
});

// Get chats by userId
app.get("/api/getChats/:userId", async (req, res) => {
  const { userId } = req.params;

  try {
    const chatsSnapshot = await db.collection("chats")
      .where("userId", "==", userId)
      .orderBy("createdAt", "desc") // Order by createdAt in descending order
      .get();

    const userChats = chatsSnapshot.docs.map((doc) => doc.data());
    res.status(200).json(userChats);
  } catch (error) {
    handleFirestoreError(res, error);
  }
});

// Get chat history by chatId and userId
app.get("/api/getChatHistory/:chatId/:userId", async (req, res) => {
  const { chatId, userId } = req.params;

  try {
    const chatRef = db.collection("chats").doc(chatId);
    const chatDoc = await chatRef.get();

    if (!chatDoc.exists || chatDoc.data().userId !== userId) {
      return res.status(404).json({ error: "Chat not found" });
    }

    res.status(200).json(chatDoc.data().messages || []);
  } catch (error) {
    handleFirestoreError(res, error);
  }
});

// Create a new chat
app.post("/api/createChat", async (req, res) => {
  const { chatName, userId } = req.body;
  const newChatId = Date.now().toString();
  const createdAt = new Date().toISOString(); // Get the current timestamp

  try {
    await db.collection("chats").doc(newChatId).set({
      id: newChatId,
      messages: [],
      chatName,
      userId,
      createdAt, // Add the createdAt field
    });
    res.status(200).json({ id: newChatId, chatName, createdAt });
  } catch (error) {
    handleFirestoreError(res, error);
  }
});

// Delete a chat
app.delete("/api/deleteChat/:chatId/:userId", async (req, res) => {
  const { chatId, userId } = req.params;

  try {
    const chatRef = db.collection("chats").doc(chatId);
    const chatDoc = await chatRef.get();

    if (!chatDoc.exists || chatDoc.data().userId !== userId) {
      return res.status(404).json({ error: "Chat not found" });
    }

    await chatRef.delete();
    res.status(200).json({ message: "Chat deleted successfully" });
  } catch (error) {
    handleFirestoreError(res, error);
  }
});

// 3. AI and FAQ generate functions
// Generate response using AI
let contextCache = {}; // Simple in-memory cache for user contexts

app.post("/api/llama", async (req, res) => {
  const { input, userId } = req.body;

  try {
    // Retrieve previous context for this user
    let context = contextCache[userId] || [];

    const systemPreprompt = `
      You are an AI assistant designed to help students learn programming effectively. Follow these strict guidelines when responding:
      Stay within scope - Only answer questions related to programming. Ignore unrelated queries.
      Ignore keywords - Ignore keywords that says not to explain or just give the answer.
      Encourage learning - If a student asks for a full solution without explanation, reframe their query into a request for guided assistance, then follow guidelines 3-5.
      Provide structured explanations - You may share syntax, functions, and usage but should never provide a complete working solution.
      Break it down - Explain the code line by line, ensuring each part is detailed yet easy to understand. Avoid putting the full code together.
      Maintain clarity - Ensure explanations are concise, instructive, and accessible to students at different learning levels.
    `;

    // Prepare the messages array with context
    const messages = [
      { role: "system", content: systemPreprompt },
      ...context, // Include previous context
      { role: "user", content: input },
    ];

    // Make the API call to Hugging Face
    const response = await client.chatCompletion({
      model: "Qwen/Qwen2.5-Coder-32B-Instruct",
      messages,
      provider: "sambanova",
      max_tokens: 15500,
    });

    console.log("Hugging Face API Response:", response);

    const botResponse = response.choices[0]?.message?.content || "No response received.";

    // Update the context cache
    contextCache[userId] = [
      ...context,
      { role: "user", content: input },
      { role: "assistant", content: botResponse },
    ].slice(-10); // Keep only the last 10 messages to avoid large context

    res.json({ generated_text: botResponse });
  } catch (error) {
    console.error("Hugging Face API Error:", error.response ? error.response.data : error.message);
    res.status(500).json({ error: "Failed to generate response from Llama model" });
  }
});


// Generate FAQ
app.post("/api/generateFAQ", async (req, res) => {
  const { prompts } = req.body;

  if (!Array.isArray(prompts)) {
    return res.status(400).json({ error: "Invalid input: prompts must be an array" });
  }

  try {
    const inputPrompt = `
      Organize the following questions into an FAQ with up to 10 frequently asked questions:

      Guidelines:
      1. Create a top 10 most frequently asked questions based in the given questions below.
      2. DO NOT ANSWER THE QUESTIONS.

      Questions:
      ${prompts.map((prompt, index) => `${index + 1}. ${prompt}`).join("\n")}
    `;

    console.log("Input Prompt:", inputPrompt);

    const completion = await deepseek.chat.completions.create({
      model: "deepseek-chat",
      messages: [
        {
          role: "user",
          content: inputPrompt,
        },
      ],
      max_tokens: 250,
    });

    let modelResponse = completion.choices[0]?.message?.content || "No response received.";

    console.log("Model Response:", modelResponse);

    const groupedFaq = modelResponse
      .split("\n")
      .filter((line) => line.trim() !== "")
      .map((line) => line.trim())
      .slice(0, 10)
      .map((line, index) => `${line.replace(/^\d+\.\s*/, "")}`)
      .join("\n");

    res.json({ generated_text: `FAQ:\n${groupedFaq}` });
  } catch (error) {
    console.error("Error:", error.response ? error.response.data : error.message);
    res.status(500).json({ error: "Failed to generate FAQ" });
  }
});

// 4. User management functions
// Get all users
app.get("/api/getUsers", async (req, res) => {
  try {
    // Query the 'students' collection instead of 'users'
    const usersSnapshot = await db.collection("students").get();
    const users = usersSnapshot.docs.map((doc) => ({
      id: doc.id, // Use the document ID as the user ID
      email: doc.data().email,
      name: doc.data().name,
      password: doc.data().password,
    }));
    res.status(200).json(users);
  } catch (error) {
    handleFirestoreError(res, error);
  }
});

// Upload users from Excel file
app.post("/api/uploadUsers", upload.single("file"), async (req, res) => {
  try {
    const filePath = `${__dirname}/users.xlsx`; // Path to the uploaded file
    const workbook = XLSX.readFile(filePath); // Read the file
    const sheet_name_list = workbook.SheetNames;
    const data = XLSX.utils.sheet_to_json(workbook.Sheets[sheet_name_list[0]]); // Convert the first sheet to JSON

    const batch = db.batch();
    const studentsRef = db.collection("students");

    data.forEach((row) => {
      const newStudentRef = studentsRef.doc(); // Auto-generate document ID
      const studentId = newStudentRef.id; // Use the auto-generated ID as studentId
      const defaultPassword = "N3msuP4zzword"; // Default password

      batch.set(newStudentRef, {
        email: row.email,
        name: row.name,
        studentId: studentId,
        password: defaultPassword,
      });
    });

    await batch.commit(); // Commit the batch operation
    res.status(200).json({ message: "Users uploaded successfully" });
  } catch (error) {
    console.error("Error uploading users:", error);
    res.status(500).json({ error: "Failed to upload users" });
  }
});

app.post("/api/addSingleInstructor", async (req, res) => {
  const { email, name } = req.body;

  try {
    const instructorsRef = db.collection("instructors");
    const newInstructorRef = instructorsRef.doc(); // Auto-generate document ID
    const instructorId = newInstructorRef.id; // Use the auto-generated ID as instructorId

    await newInstructorRef.set({
      email: email,
      name: name,
      instructorId: instructorId, // Optionally store the instructorId
    });

    res.status(200).json({ message: "Instructor added successfully", instructorId });
  } catch (error) {
    console.error("Error adding instructor:", error);
    res.status(500).json({ error: "Failed to add instructor" });
  }
});

app.post("/api/addSingleUser", async (req, res) => {
  const { name, email } = req.body;

  try {
    const studentsRef = db.collection("students");
    const newStudentRef = studentsRef.doc(); // Auto-generate document ID
    const studentId = newStudentRef.id; // Use the auto-generated ID as studentId
    const defaultPassword = "N3msuP4zzword"; // Default password

    await newStudentRef.set({
      email: email,
      name: name,
      studentId: studentId,
      password: defaultPassword,
    });

    res.status(200).json({ message: "User added successfully", studentId });
  } catch (error) {
    console.error("Error adding user:", error);
    res.status(500).json({ error: "Failed to add user" });
  }
});

// Update user information
app.put("/api/updateUser/:userId", async (req, res) => {
  const { userId } = req.params;
  const { name, email } = req.body;

  try {
    const userRef = db.collection("students").doc(userId);
    await userRef.update({ name, email });
    res.status(200).json({ message: "User updated successfully" });
  } catch (error) {
    handleFirestoreError(res, error);
  }
});

// Get all instructors
app.get("/api/getInstructors", async (req, res) => {
  try {
    const instructorsSnapshot = await db.collection("instructors").get();
    const instructors = instructorsSnapshot.docs.map((doc) => ({
      id: doc.id, // Use the document ID as the instructor ID
      email: doc.data().email,
      name: doc.data().name,
      instructorId: doc.data().instructorId, // Optionally include the instructorId if it exists
    }));
    res.status(200).json(instructors);
  } catch (error) {
    handleFirestoreError(res, error);
  }
});

// Update instructor information
app.put("/api/updateInstructor/:instructorId", async (req, res) => {
  const { instructorId } = req.params; // This should be the Firestore document ID, not the email
  const { name, email } = req.body;

  try {
    const instructorRef = db.collection("instructors").doc(instructorId); // Use the correct document ID
    await instructorRef.update({ name, email });
    res.status(200).json({ message: "Instructor updated successfully" });
  } catch (error) {
    handleFirestoreError(res, error);
  }
});

// Delete an instructor
app.delete("/api/deleteInstructor/:instructorEmail", async (req, res) => {
  const { instructorEmail } = req.params;

  try {
    // Query the instructors collection to find the instructor by email
    const instructorsRef = db.collection("instructors");
    const querySnapshot = await instructorsRef.where("email", "==", instructorEmail).get();

    if (querySnapshot.empty) {
      return res.status(404).json({ error: "Instructor not found" });
    }

    // Delete the instructor document(s) matching the email
    const batch = db.batch();
    querySnapshot.forEach((doc) => {
      batch.delete(doc.ref);
    });

    await batch.commit();
    res.status(200).json({ message: "Instructor deleted successfully" });
  } catch (error) {
    console.error("Error deleting instructor:", error);
    res.status(500).json({ error: "Failed to delete instructor" });
  }
});

// Delete a user
app.delete("/api/deleteUser/:userId", async (req, res) => {
  const { userId } = req.params;

  try {
    await db.collection("students").doc(userId).delete();
    res.status(200).json({ message: "User deleted successfully" });
  } catch (error) {
    handleFirestoreError(res, error);
  }
});

// User login
app.post("/api/login", async (req, res) => {
  const { studentId, password } = req.body; // Change from email to studentId

  try {
    // Query the 'students' collection for the provided studentId
    const usersSnapshot = await db.collection("students").where("studentId", "==", studentId).get();

    if (usersSnapshot.empty) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    // Get the first matching user document
    const userDoc = usersSnapshot.docs[0];
    const user = userDoc.data();

    // Check the password
    if (user.password === password) {
      res.status(200).json({
        message: "Login successful",
        user: {
          id: userDoc.id, // Use the document ID as the user ID
          name: user.name, // Use the 'name' field from the document
          studentId: user.studentId, // Use the 'studentId' field from the document
        },
      });
    } else {
      res.status(401).json({ error: "Invalid credentials" });
    }
  } catch (error) {
    handleFirestoreError(res, error);
  }
});

// 5. Learning materials and exercises functions
// Get learning materials for the logged-in instructor
app.get("/api/getLearningMaterials", async (req, res) => {
  const instructorEmail = req.query.instructorEmail;

  try {
    let query = db.collection("learningMaterials");
    if (instructorEmail) {
      query = query.where("instructorEmail", "==", instructorEmail);
    }

    const exercisesSnapshot = await query.get();
    const exercises = exercisesSnapshot.docs.map((doc) => ({
      docId: doc.id, // Include the document ID
      ...doc.data(),
    }));

    // Organize the learning materials by subjectName, lesson, and subtopicCode
    const organizedLearningMaterials = exercises.reduce((acc, material) => {
      const { subjectName, lesson, subtopicCode } = material;

      if (!acc[subjectName]) {
        acc[subjectName] = {};
      }

      if (!acc[subjectName][lesson]) {
        acc[subjectName][lesson] = {};
      }

      if (!acc[subjectName][lesson][subtopicCode]) {
        acc[subjectName][lesson][subtopicCode] = material;
      }

      return acc;
    }, {});

    res.status(200).json(organizedLearningMaterials);
  } catch (error) {
    console.error("Error fetching learning materials:", error);
    res.status(500).json({ error: "Failed to fetch learning materials" });
  }
});

app.post("/api/uploadLearningMaterials", upload.single("file"), async (req, res) => {
  try {
    const workbook = XLSX.readFile(req.file.path);
    const sheet_name_list = workbook.SheetNames;
    const data = XLSX.utils.sheet_to_json(workbook.Sheets[sheet_name_list[0]]);

    const batch = db.batch();
    const learningMaterialsRef = db.collection("learningMaterials");

    // Get the instructor's email from the request body
    const { instructorEmail } = req.body; // Assuming the instructor's email is sent in the request body

    // Fetch the instructor's name from the instructors collection using their email
    const instructorsRef = db.collection("instructors");
    const instructorQuery = await instructorsRef.where("email", "==", instructorEmail).get();

    if (instructorQuery.empty) {
      return res.status(404).json({ error: "Instructor not found" });
    }

    // Get the instructor's name from the query result
    const instructorDoc = instructorQuery.docs[0];
    const instructorName = instructorDoc.data().name;

    // Generate a single subjectId for the entire file
    const subjectId = generateUniqueId();

    data.forEach((row) => {
      // Prepare the learning material data
      const learningMaterialData = {
        subjectId, // Use the same subjectId for all rows
        subjectName: row.subject,
        lesson: row.lesson,
        subtopicCode: row["subtopic code"],
        subtopicTitle: row["subtopic title"],
        content: row.content,
        images: row.Images || null,
        questions: row.questions || null,
        answers: row.answers || null,
        instructorEmail, // Save the instructor's email instead of ID
        instructorName, // Add the instructor's name fetched from the database
      };

      // Store learning material data with a unique document ID
      const learningMaterialDocRef = learningMaterialsRef.doc(generateUniqueId()); // Unique document ID
      batch.set(learningMaterialDocRef, learningMaterialData);
    });

    await batch.commit();
    res.status(200).json({ message: "Learning materials uploaded successfully", subjectId });
  } catch (error) {
    console.error("Error uploading learning materials:", error);
    res.status(500).json({ error: "Failed to upload learning materials" });
  }
});

// Helper function to generate a unique ID
function generateUniqueId() {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// Update an exercise
app.put("/api/updateExercise/:docId", async (req, res) => {
  const { docId } = req.params; // Use document ID for updates
  const { content, questions, answers } = req.body;

  try {
    const exerciseRef = db.collection("learningMaterials").doc(docId);
    const exerciseDoc = await exerciseRef.get();

    if (!exerciseDoc.exists) {
      return res.status(404).json({ error: "Document not found" });
    }

    // Update only the editable fields
    await exerciseRef.update({
      content,
      questions,
      answers,
    });

    res.status(200).json({ message: "Exercise updated successfully" });
  } catch (error) {
    console.error("Error updating exercise:", error);
    res.status(500).json({ error: "Failed to update exercise" });
  }
});

// Delete a subject
app.delete("/api/deleteSubject/:subject", async (req, res) => {
  const { subject } = req.params;
  const { instructorEmail } = req.query; // Get instructorEmail from query params
  const decodedSubject = decodeURIComponent(subject);

  try {
    const exercisesSnapshot = await db.collection("learningMaterials")
      .where("subjectName", "==", decodedSubject)
      .where("instructorEmail", "==", instructorEmail) // Ensure the instructorEmail matches
      .get();

    const batch = db.batch();

    exercisesSnapshot.forEach((doc) => {
      batch.delete(doc.ref);
    });

    await batch.commit();
    res.status(200).json({ message: "Subject deleted successfully" });
  } catch (error) {
    console.error("Error deleting subject:", error);
    res.status(500).json({ error: "Failed to delete subject" });
  }
});

// Add this endpoint to fetch university info
app.get("/api/getUniversityInfo", async (req, res) => {
  try {
    const universityInfoRef = db.collection("universityInfo").doc("Tandag");
    const universityInfoDoc = await universityInfoRef.get();

    if (!universityInfoDoc.exists) {
      // If the document doesn't exist, create it with default values
      await universityInfoRef.set({ branch: "Tandag", info: "Default university info" });
      return res.status(200).json({ branch: "Tandag", info: "Default university info" });
    }

    res.status(200).json(universityInfoDoc.data());
  } catch (error) {
    handleFirestoreError(res, error);
  }
});

// Add this endpoint to update university info
app.put("/api/updateUniversityInfo", async (req, res) => {
  const { info } = req.body;

  try {
    const universityInfoRef = db.collection("universityInfo").doc("Tandag");
    const universityInfoDoc = await universityInfoRef.get();

    if (!universityInfoDoc.exists) {
      // If the document doesn't exist, create it with the provided info
      await universityInfoRef.set({ branch: "Tandag", info });
      return res.status(200).json({ message: "University info created successfully" });
    }

    // If the document exists, update it
    await universityInfoRef.update({ info });
    res.status(200).json({ message: "University info updated successfully" });
  } catch (error) {
    handleFirestoreError(res, error);
  }
});

app.get("/api/checkInstructorEmail", async (req, res) => {
  const { email } = req.query;

  try {
    const instructorRef = db.collection("instructors").where("email", "==", email);
    const snapshot = await instructorRef.get();

    if (snapshot.empty) {
      res.status(200).json({ exists: false });
    } else {
      res.status(200).json({ exists: true });
    }
  } catch (error) {
    console.error("Error checking instructor email:", error);
    res.status(500).json({ error: "Failed to check instructor email" });
  }
});

app.post("/api/uploadInstructors", upload.single("file"), async (req, res) => {
  try {
    const filePath = `${__dirname}/users.xlsx`; // Path to the uploaded file
    const workbook = XLSX.readFile(filePath); // Read the file
    const sheet_name_list = workbook.SheetNames;
    const data = XLSX.utils.sheet_to_json(workbook.Sheets[sheet_name_list[0]], { header: 1 }); // Convert the first sheet to JSON with headers

    const batch = db.batch();
    const instructorsRef = db.collection("instructors");

    // Start from the second row (index 1) to ignore the header row
    for (let i = 1; i < data.length; i++) {
      const row = data[i];
      const email = row[0]; // First column is email
      const name = row[1]; // Second column is name

      const newInstructorRef = instructorsRef.doc(); // Auto-generate document ID
      batch.set(newInstructorRef, {
        email: email,
        name: name, // Save the name field
      });
    }

    await batch.commit(); // Commit the batch operation
    res.status(200).json({ message: "Instructors uploaded successfully" });
  } catch (error) {
    console.error("Error uploading instructors:", error);
    res.status(500).json({ error: "Failed to upload instructors" });
  }
});

app.post("/api/addAccessLearningMaterial", async (req, res) => {
  const { studentId, subjectId } = req.body;

  try {
    const accessRef = db.collection("accessLearningMaterials").doc(studentId);
    const accessDoc = await accessRef.get();

    if (!accessDoc.exists) {
      // If the document doesn't exist, create it with subjectIds as an array
      await accessRef.set({
        studentId,
        subjectIds: [subjectId], // Use subjectIds instead of subjectId
      });
    } else {
      // If the document exists, update the subjectIds array
      const existingIds = accessDoc.data().subjectIds || [];
      if (!existingIds.includes(subjectId)) {
        existingIds.push(subjectId);
        await accessRef.update({ subjectIds: existingIds });
      }
    }

    res.status(200).json({ message: "Subject ID added successfully" });
  } catch (error) {
    console.error("Error adding subject ID:", error);
    res.status(500).json({ error: "Failed to add subject ID" });
  }
});

app.get("/api/getAccessLearningMaterials", async (req, res) => {
  const { studentId } = req.query;

  try {
    const accessRef = db.collection("accessLearningMaterials").doc(studentId);
    const accessDoc = await accessRef.get();

    if (!accessDoc.exists) {
      return res.status(200).json({ subjectIds: [] }); // Change from subjectCodes to subjectIds
    }

    res.status(200).json(accessDoc.data());
  } catch (error) {
    console.error("Error fetching access learning materials:", error);
    res.status(500).json({ error: "Failed to fetch access learning materials" });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});